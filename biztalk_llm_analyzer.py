#!/usr/bin/env python3
"""
BizTalk LLM Analyzer

Extracts content from BizTalk components and uses LLM to generate business requirements
based solely on actual component contents without assumptions.
"""

import os
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
from dotenv import load_dotenv
from llm_json_parser import LLMJSONParser
load_dotenv()  
print("Environment variables loaded from .env file")
# Import LLM utilities from the project
from resilient_llm_caller import get_llm_caller
from llm_token_tracker import create_tracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("BizTalkLLMAnalyzer")

# BizTalk file extensions and their types
BIZTALK_FILE_TYPES = {
    ".btm": "map",
    ".btp": "pipeline", 
    ".xsd": "schema",
    ".odx": "orchestration",
    ".xsl": "xslt",
    ".xslt": "xslt",
    ".xml": "xml",
    ".config": "config"
}

class BizTalkLLMAnalyzer:
    """Analyzes BizTalk components using LLM to extract business requirements."""
    
    def __init__(self, input_dir: str, output_dir: str, cache_dir: str = None):
        """
        Initialize the analyzer.
        
        Args:
            input_dir: Directory containing BizTalk components
            output_dir: Directory for output
            cache_dir: Directory for caching LLM responses (defaults to output_dir/cache)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir or os.path.join(output_dir, "cache")
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize component collections
        self.components = {
            "maps": [],
            "pipelines": [],
            "schemas": [],
            "orchestrations": [],
            "xslts": [],
            "other": []
        }
        
        # Setup LLM and token tracking
        self.llm = get_llm_caller()
        self.token_tracker = create_tracker("biztalk_analyzer")
        
        logger.info(f"BizTalk LLM Analyzer initialized")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def scan_biztalk_folder(self):
        """Scan the input directory for BizTalk components."""
        logger.info(f"Scanning for BizTalk components in {self.input_dir}")
        
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # Skip non-BizTalk files
                if file_ext not in BIZTALK_FILE_TYPES:
                    continue
                
                # Categorize by file type
                component_type = BIZTALK_FILE_TYPES[file_ext]
                component_info = {
                    "name": file,
                    "path": file_path,
                    "type": component_type
                }
                
                # Add to appropriate collection
                if component_type == "map":
                    self.components["maps"].append(component_info)
                elif component_type == "pipeline":
                    self.components["pipelines"].append(component_info)
                elif component_type == "schema":
                    self.components["schemas"].append(component_info)
                elif component_type == "orchestration":
                    self.components["orchestrations"].append(component_info)
                elif component_type == "xslt":
                    self.components["xslts"].append(component_info)
                else:
                    self.components["other"].append(component_info)
        
        # Log summary
        total_components = sum(len(comps) for comps in self.components.values())
        logger.info(f"Found {total_components} BizTalk components:")
        for comp_type, comps in self.components.items():
            if comps:
                logger.info(f"  - {len(comps)} {comp_type}")
    
    def extract_component_content(self, component: Dict[str, str]) -> str:
        """
        Extract content from a BizTalk component file.
        
        Args:
            component: Component info dictionary with name, path, and type
            
        Returns:
            Raw content of the component file
        """
        try:
            with open(component["path"], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading {component['path']}: {e}")
            return ""
    
    def process_component_with_llm(self, component: Dict[str, str], content: str) -> Dict[str, Any]:
        """
        Process a component's content with LLM.
        
        Args:
            component: Component info dictionary
            content: Raw content of the component
            
        Returns:
            Dictionary with LLM analysis results
        """
        # Check if cached result exists
        cache_file = os.path.join(
            self.cache_dir, 
            f"{component['type']}_{re.sub(r'[^\w\-\.]', '_', component['name'])}.json"
        )
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                logger.warning(f"Could not read cache file {cache_file}, will reprocess")
        
        # Create prompt based on component type
        prompt = self._create_component_prompt(component, content)
        
        # Call LLM with resilient caller
        logger.info(f"Processing {component['name']} with LLM")
        llm_response = self.llm.call_with_retry(
            prompt=prompt,
            temperature=0.2  # Low temperature for factual extraction
        )
        
        if not llm_response:
            logger.error(f"Failed to get LLM response for {component['name']}")
            return {"error": "LLM processing failed"}
        
        # Extract JSON from response
        try:
            result = json.loads(llm_response)
        except json.JSONDecodeError:    
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', llm_response)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    # If still failing, extract just the text as a fallback
                    result = {"raw_analysis": llm_response}
            else:
                # Try to find anything that looks like JSON
                json_match = re.search(r'(\{[\s\S]*\})', llm_response)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        result = {"raw_analysis": llm_response}
                else:
                    result = {"raw_analysis": llm_response}
        
        # Add metadata
        result["component_name"] = component["name"]
        result["component_type"] = component["type"]
        result["timestamp"] = time.time()
        
        # Cache result
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def _create_component_prompt(self, component: Dict[str, str], content: str) -> str:
        """Create an appropriate prompt for the component type."""
        base_prompt = (
            f"Analyze the following BizTalk {component['type']} component named '{component['name']}' "
            f"and extract business requirements, transformations, rules, and patterns SOLELY based "
            f"on the actual code/content provided. DO NOT make assumptions or add information not "
            f"present in the component.\n\n"
            f"The component content is:\n```\n{content}\n```\n\n"
        )
        
        if component["type"] == "map":
            prompt = base_prompt + (
                "Focus on:\n"
                "1. Source and target schemas/systems\n"
                "2. Field mappings and transformations\n"
                "3. Any business rules implemented in functoids\n"
                "4. Data formats and structures\n"
            )
        elif component["type"] == "schema":
            prompt = base_prompt + (
                "Focus on:\n"
                "1. Entity structure and data model\n"
                "2. Field definitions, types, and constraints\n"
                "3. Namespaces and document types\n"
                "4. Any annotations indicating business context\n"
            )
        elif component["type"] == "pipeline":
            prompt = base_prompt + (
                "Focus on:\n"
                "1. Pipeline stages and components\n"
                "2. Message processing operations\n"
                "3. Validation, transformation, or encoding steps\n"
                "4. Error handling mechanisms\n"
            )
        elif component["type"] == "orchestration":
            prompt = base_prompt + (
                "Focus on:\n"
                "1. Business process flow\n"
                "2. Decision logic and branching\n"
                "3. Error handling and compensation\n"
                "4. Integration points and systems\n"
            )
        elif component["type"] == "xslt":
            prompt = base_prompt + (
                "Focus on:\n"
                "1. Source to target field mappings\n"
                "2. Transformation logic and rules\n"
                "3. Conditional processing and business rules\n"
                "4. Data formats and structure changes\n"
            )
        else:
            prompt = base_prompt + (
                "Extract any relevant business requirements or technical details from this component."
            )
        
        # Add output format instructions
        prompt += (
            "\n\nProvide your analysis in JSON format with these sections:\n"
            "- summary: Brief overview of the component's purpose\n"
            "- business_entities: Business entities identified\n"
            "- business_rules: Business rules implemented\n"
            "- transformations: Data transformations\n"
            "- integration_points: Systems or components integrated\n"
            "- technical_details: Technical implementation details relevant for ACE migration\n"
            "- fields: List of fields processed (with source and target mappings if applicable)\n\n"
            "Only include sections that have actual content from the component. If information "
            "is not present in the component, leave the section empty. Do not make assumptions."
                "\n\nIMPORTANT: Format your entire response as a valid JSON object. Do not include any text, explanation, or markdown formatting outside the JSON. Start with '{' and end with '}' without any additional text. For example:\n\n"
            "{\n"
            "  \"summary\": \"This component...\",\n"
            "  \"business_entities\": [...],\n"
            "  \"business_rules\": [...]\n"
            "}\n\n"
            "Do not include ```json or ``` markers around the response."
        )
        
        return prompt
    
    def process_all_components(self):
        """Process all BizTalk components with LLM."""
        results = {
            "maps": [],
            "pipelines": [],
            "schemas": [],
            "orchestrations": [],
            "xslts": [],
            "other": []
        }
        
        # Process each component type
        for comp_type, components in self.components.items():
            logger.info(f"Processing {len(components)} {comp_type} components")
            
            for component in components:
                # Extract content
                content = self.extract_component_content(component)
                if not content:
                    continue
                
                # Process with LLM
                result = self.process_component_with_llm(component, content)
                results[comp_type].append(result)
        
        # Save all results
        results_file = os.path.join(self.output_dir, "component_analysis.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved component analysis to {results_file}")
        return results
    



    def parse_raw_analysis(self, raw_analysis):
        """
        Parse the raw_analysis field to extract structured information
        
        Args:
            raw_analysis: String containing LLM analysis
            
        Returns:
            Dictionary with extracted information
        """
        parser = LLMJSONParser(debug=False)
        result = parser.parse(raw_analysis)
        
        # If JSON parsing succeeded, return the data
        if result.success:
            return result.data
        
        # If not JSON, try to extract sections using regex patterns
        extracted_data = {
            "business_entities": [],
            "business_rules": [],
            "transformations": [], 
            "patterns": []
        }
        
        # Look for business requirements sections with various formats
        section_patterns = [
            # Format: "Business Requirements: - item1 - item2"
            (r'(?:Business Requirements?|I\.\s*Business Requirements?)[:\n]+((?:\s*[-•*]\s*[^\n]+\n*)+)', 'business_entities'),
            
            # Format: "Rules: - rule1 - rule2" 
            (r'(?:Rules?|III\.\s*Rules?)[:\n]+((?:\s*[-•*]\s*[^\n]+\n*)+)', 'business_rules'),
            
            # Format: "Transformations: - trans1 - trans2"
            (r'(?:Transformations?|II\.\s*Transformations?)[:\n]+((?:\s*[-•*]\s*[^\n]+\n*)+)', 'transformations'),
            
            # Format: "Patterns: - pattern1 - pattern2"
            (r'(?:Patterns?|IV\.\s*Patterns?)[:\n]+((?:\s*[-•*]\s*[^\n]+\n*)+)', 'patterns')
        ]
        
        for pattern, key in section_patterns:
            matches = re.search(pattern, raw_analysis, re.IGNORECASE | re.MULTILINE)
            if matches:
                items_text = matches.group(1)
                # Extract list items
                items = re.findall(r'[-•*]\s*([^\n]+)', items_text)
                extracted_data[key] = [item.strip() for item in items if item.strip()]
        
        # Extract integration points using pattern matching
        integration_pattern = r'(?:Integration Points?|External Systems?|References?)[:\n]+((?:\s*[-•*]\s*[^\n]+\n*)+)'
        matches = re.search(integration_pattern, raw_analysis, re.IGNORECASE | re.MULTILINE)
        if matches:
            items_text = matches.group(1)
            integration_points = re.findall(r'[-•*]\s*([^\n]+)', items_text)
            extracted_data["integration_points"] = [point.strip() for point in integration_points if point.strip()]
        else:
            extracted_data["integration_points"] = []
        
        # Extract message formats
        message_pattern = r'(?:Message Formats?|Data Formats?|Schema)[:\n]+((?:\s*[-•*]\s*[^\n]+\n*)+)'
        matches = re.search(message_pattern, raw_analysis, re.IGNORECASE | re.MULTILINE)
        if matches:
            items_text = matches.group(1)
            formats = re.findall(r'[-•*]\s*([^\n]+)', items_text)
            extracted_data["message_formats"] = [fmt.strip() for fmt in formats if fmt.strip()]
        else:
            extracted_data["message_formats"] = []
        
        # Try to generate a summary if it doesn't exist
        if "summary" not in extracted_data or not extracted_data["summary"]:
            # First look for a summary section
            summary_match = re.search(r'(?:Summary|Overview)[:\n]+\s*([^\n]+)', raw_analysis, re.IGNORECASE)
            if summary_match:
                extracted_data["summary"] = summary_match.group(1).strip()
            else:
                # Extract first sentence as summary
                first_sentence = re.search(r'^([^.!?\n]+[.!?])', raw_analysis)
                if first_sentence:
                    extracted_data["summary"] = first_sentence.group(1).strip()
                else:
                    # Take first 100 chars as summary if nothing else available
                    extracted_data["summary"] = raw_analysis[:100].strip() + "..."
        
        return extracted_data

    def generate_business_requirements(self, analysis_results: Dict[str, List[Dict[str, Any]]]):
        """
        Generate business requirements document from component analyses.
        
        Args:
            analysis_results: Dictionary of component analysis results
        """
        logger.info("Generating business requirements document")
        
        # Prepare consolidated data
        consolidated = {
            "business_entities": set(),
            "business_rules": [],
            "transformations": [],
            "integration_points": set(),
            "message_formats": set(),
            "technical_components": {}
        }
        
        # Process each component's analysis
        for comp_type, results in analysis_results.items():
            for result in results:
                # Skip entries with errors
                if "error" in result:
                    continue
                
                # Add component to technical components
                component_name = result.get("component_name", "Unknown")
                component_type = result.get("component_type", "unknown")
                
                if component_type not in consolidated["technical_components"]:
                    consolidated["technical_components"][component_type] = []
                
                # Extract information from raw_analysis if it exists
                if "raw_analysis" in result:
                    extracted_data = self.parse_raw_analysis(result["raw_analysis"])
                    
                    # Update summary if available
                    component_summary = extracted_data.get("summary", result.get("summary", ""))
                    
                    # Add to technical components with summary
                    consolidated["technical_components"][component_type].append({
                        "name": component_name,
                        "summary": component_summary
                    })
                    
                    # Add business entities
                    entities = extracted_data.get("business_entities", [])
                    if isinstance(entities, list):
                        consolidated["business_entities"].update(entities)
                    
                    # Add business rules
                    rules = extracted_data.get("business_rules", [])
                    if isinstance(rules, list):
                        for rule in rules:
                            if rule and rule not in consolidated["business_rules"]:
                                consolidated["business_rules"].append(rule)
                    
                    # Add transformations
                    transforms = extracted_data.get("transformations", [])
                    if isinstance(transforms, list):
                        for transform in transforms:
                            if transform and transform not in consolidated["transformations"]:
                                consolidated["transformations"].append(transform)
                    
                    # Add integration points
                    points = extracted_data.get("integration_points", [])
                    if isinstance(points, list):
                        consolidated["integration_points"].update(points)
                    
                    # Add message formats
                    formats = extracted_data.get("message_formats", [])
                    if isinstance(formats, list):
                        consolidated["message_formats"].update(formats)
                    
                else:
                    # Fallback to existing fields if raw_analysis isn't available
                    consolidated["technical_components"][component_type].append({
                        "name": component_name,
                        "summary": result.get("summary", "")
                    })
                    
                    # Add business entities
                    entities = result.get("business_entities", [])
                    if isinstance(entities, list):
                        consolidated["business_entities"].update(entities)
                    
                    # Add business rules
                    rules = result.get("business_rules", [])
                    if isinstance(rules, list):
                        for rule in rules:
                            if rule and rule not in consolidated["business_rules"]:
                                consolidated["business_rules"].append(rule)
                    
                    # Add transformations
                    transforms = result.get("transformations", [])
                    if isinstance(transforms, list):
                        for transform in transforms:
                            if transform and transform not in consolidated["transformations"]:
                                consolidated["transformations"].append(transform)
                    
                    # Add integration points
                    points = result.get("integration_points", [])
                    if isinstance(points, list):
                        consolidated["integration_points"].update(points)
                    
                    # Extract message formats from fields
                    fields = result.get("fields", [])
                    if isinstance(fields, list):
                        for field in fields:
                            if isinstance(field, dict):
                                source_format = field.get("source_format")
                                target_format = field.get("target_format")
                                if source_format:
                                    consolidated["message_formats"].add(source_format)
                                if target_format:
                                    consolidated["message_formats"].add(target_format)
        
        # Convert sets to lists for JSON serialization
        consolidated["business_entities"] = list(consolidated["business_entities"])
        consolidated["integration_points"] = list(consolidated["integration_points"])
        consolidated["message_formats"] = list(consolidated["message_formats"])
        
        # Save consolidated data
        consolidated_file = os.path.join(self.output_dir, "consolidated_analysis.json")
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        
        # Generate HTML document
        self._generate_html_document(consolidated)


    
    def _generate_html_document(self, consolidated: Dict[str, Any]):
        """Generate HTML business requirements document from consolidated data."""
        # Basic HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>BizTalk Business Requirements Document</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #2c3e50;
                    border-bottom: 1px solid #bdc3c7;
                    padding-bottom: 5px;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #2c3e50;
                    margin-top: 25px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    text-align: left;
                    padding: 8px;
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .metadata {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    font-size: 0.9em;
                }}
                .code {{
                    font-family: monospace;
                    background-color: #f7f7f7;
                    padding: 10px;
                    border-radius: 3px;
                    overflow-x: auto;
                }}
                .note {{
                    background-color: #e7f4fe;
                    border-left: 4px solid #3498db;
                    padding: 10px;
                    margin: 15px 0;
                }}
                .empty-section {{
                    color: #777;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <h1>BizTalk Business Requirements Document</h1>
            
            <div class="metadata">
                <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>BizTalk Folder:</strong> {self.input_dir}</p>
            </div>
            
            <h2>1. Executive Summary</h2>
            
            <p>This document outlines the business requirements for the integration system 
            identified in the BizTalk components. The analysis was performed using LLM to extract
            information directly from the component content without external assumptions.</p>
            
            <h3>Component Summary:</h3>
            <ul>
        """
        
        # Add component counts
        for comp_type, components in self.components.items():
            if components:
                html_content += f"<li><strong>{comp_type.title()}:</strong> {len(components)}</li>\n"
        
        html_content += """
            </ul>
            
            <h2>2. Business Entities</h2>
        """
        
        # Add business entities
        if consolidated["business_entities"]:
            html_content += "<ul>\n"
            for entity in sorted(consolidated["business_entities"]):
                html_content += f"<li>{entity}</li>\n"
            html_content += "</ul>\n"
        else:
            html_content += "<p class='empty-section'>No business entities could be identified from the components.</p>\n"
        
        # Add business rules
        html_content += "<h2>3. Business Rules</h2>\n"
        
        if consolidated["business_rules"]:
            html_content += "<ul>\n"
            for rule in consolidated["business_rules"]:
                html_content += f"<li>{rule}</li>\n"
            html_content += "</ul>\n"
        else:
            html_content += "<p class='empty-section'>No business rules could be identified from the components.</p>\n"
        
        # Add message formats
        html_content += "<h2>4. Message Formats</h2>\n"
        
        if consolidated["message_formats"]:
            html_content += "<ul>\n"
            for format_name in sorted(consolidated["message_formats"]):
                html_content += f"<li>{format_name}</li>\n"
            html_content += "</ul>\n"
        else:
            html_content += "<p class='empty-section'>No message formats could be identified from the components.</p>\n"
        
        # Add transformations
        html_content += "<h2>5. Data Transformations</h2>\n"
        
        if consolidated["transformations"]:
            html_content += "<ul>\n"
            for transform in consolidated["transformations"]:
                html_content += f"<li>{transform}</li>\n"
            html_content += "</ul>\n"
        else:
            html_content += "<p class='empty-section'>No data transformations could be identified from the components.</p>\n"
        
        # Add integration points
        html_content += "<h2>6. Integration Points</h2>\n"
        
        if consolidated["integration_points"]:
            html_content += "<ul>\n"
            for point in sorted(consolidated["integration_points"]):
                html_content += f"<li>{point}</li>\n"
            html_content += "</ul>\n"
        else:
            html_content += "<p class='empty-section'>No integration points could be identified from the components.</p>\n"
        
        # Add technical components
        html_content += "<h2>7. Technical Components</h2>\n"
        
        for comp_type, components in consolidated["technical_components"].items():
            if components:
                html_content += f"<h3>7.{list(consolidated['technical_components'].keys()).index(comp_type) + 1} {comp_type.title()} Components</h3>\n"
                html_content += "<table>\n"
                html_content += "<tr><th>Component</th><th>Description</th></tr>\n"
                
                for component in components:
                    html_content += f"<tr><td>{component['name']}</td><td>{component['summary']}</td></tr>\n"
                
                html_content += "</table>\n"
        
        # Close the HTML document
        html_content += """
            <h2>8. Notes</h2>
            
            <p class="note">This document was generated through LLM analysis of the actual BizTalk components.
            The information presented is based solely on the content found in the components without external assumptions.</p>
            
        </body>
        </html>
        """
        
        # Write HTML to file
        
        component_name = os.path.basename(self.input_dir)
        safe_name = re.sub(r'[^\w\-\.]', '_', component_name)
        output_file = os.path.join(self.output_dir, f"{safe_name}_requirements.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated business requirements document: {output_file}")
    
    def process(self):
        """Run the complete BizTalk analysis process."""
        # Step 1: Scan for components
        self.scan_biztalk_folder()
        
        # Step 2: Process components with LLM
        analysis_results = self.process_all_components()
        
        # Step 3: Generate business requirements
        self.generate_business_requirements(analysis_results)
        
        # Step 4: Print token usage summary
        self.token_tracker.print_session_summary()
        
        logger.info("BizTalk analysis complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BizTalk LLM Analyzer")

    parser.add_argument(
        "--input_dir",
        default=r"C:\@Official\@Gen AI\DSV\BizTalk\Analyze_this_folder\Biztalk Source Code & Confluence Page_60 Flows\Biztalk Source Code & Confluence Page_60 Flows\01 - WINFRD_OUT_ShipmentUpdate_REC",  # raw string for Windows
        help="Path to BizTalk components folder",
    )

    parser.add_argument(
        "--output_dir",
        default="./business_requirement",
        help="Output directory",
    )

    args = parser.parse_args()

    analyzer = BizTalkLLMAnalyzer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    analyzer.process()

if __name__ == "__main__":
    main()


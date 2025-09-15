#!/usr/bin/env python3
"""
Smart LLM Token Tracker Module
Real-time token usage tracking and cost analysis for BizTalk-to-ACE conversion pipeline
File-based, session-focused with intelligent cost calculation and capacity planning
"""

import json
import time
import csv
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from functools import wraps
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from threading import Lock
import os


@dataclass
class TokenUsage:
    """Individual LLM call token usage record"""
    timestamp: str
    agent: str
    operation: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    execution_time_ms: int
    flow_name: Optional[str] = None


@dataclass
class SessionMetrics:
    """Real-time session metrics"""
    session_start: str
    total_calls: int
    total_tokens: int
    total_cost: float
    flows_processed: int
    average_tokens_per_flow: float
    average_cost_per_flow: float
    current_tokens_per_minute: float
    estimated_daily_capacity: int
    estimated_daily_cost: float


class SmartTokenTracker:
    """
    Smart LLM Token Tracker with real-time metrics and capacity planning
    File-based, session-focused, zero-dependency tracking
    """
    
    def __init__(self, session_name: str = None):
        self.session_name = session_name or f"session_{int(time.time())}"
        self.session_start = datetime.now()
        self.usage_records: List[TokenUsage] = []
        self.flow_count = 0
        self.lock = Lock()  # Thread-safe operations
        
        # Groq Pricing Table (USD per 1K tokens) - Updated Jan 2025
        self.pricing = {
            "llama-3.1-8b-instant": {"input": 0.000050, "output": 0.000050},
            "llama-3.1-70b-versatile": {"input": 0.000800, "output": 0.000800},
            "llama-3.3-70b-versatile": {"input": 0.000800, "output": 0.000800},
            "mixtral-8x7b-32768": {"input": 0.000240, "output": 0.000240},
            "deepseek-r1-distill-llama-70b": {"input": 0.000140, "output": 0.000280}
        }
        
        print(f"🎯 Smart Token Tracker initialized: {self.session_name}")
    
    def track_llm_call(self, agent: str, operation: str, flow_name: str = None):
        """
        Intelligent decorator for tracking LLM calls
        Usage: @tracker.track_llm_call("agent_1", "esql_generation", "flow_name")
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Execute the original LLM call
                result = func(*args, **kwargs)
                
                # Extract tokens from Groq response
                tokens_data = self._extract_tokens_from_response(result)
                execution_time = int((time.time() - start_time) * 1000)
                
                if tokens_data:
                    # Record usage
                    self._record_usage(
                        agent=agent,
                        operation=operation,
                        model=tokens_data['model'],
                        input_tokens=tokens_data['input_tokens'],
                        output_tokens=tokens_data['output_tokens'],
                        execution_time_ms=execution_time,
                        flow_name=flow_name
                    )
                
                return result
            return wrapper
        return decorator
    
    def manual_track(self, agent: str, operation: str, model: str, 
                    input_tokens: int, output_tokens: int, 
                    execution_time_ms: int = 0, flow_name: str = None):
        """Manual token tracking for cases where decorator can't be used"""
        self._record_usage(
            agent=agent,
            operation=operation, 
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            execution_time_ms=execution_time_ms,
            flow_name=flow_name
        )
    
    def _extract_tokens_from_response(self, response) -> Optional[Dict]:
        """Extract token usage from Groq API response"""
        try:
            if hasattr(response, 'usage') and response.usage:
                return {
                    'model': getattr(response, 'model', 'unknown'),
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            return None
        except Exception as e:
            print(f"⚠️ Token extraction failed: {e}")
            return None
    
    def _record_usage(self, agent: str, operation: str, model: str,
                     input_tokens: int, output_tokens: int, 
                     execution_time_ms: int, flow_name: str = None):
        """Record token usage with thread safety"""
        with self.lock:
            total_tokens = input_tokens + output_tokens
            cost = self._calculate_cost(model, input_tokens, output_tokens)
            
            # Track flow completion
            if operation in ['flow_complete', 'pipeline_complete'] or 'complete' in operation.lower():
                self.flow_count += 1
            
            usage = TokenUsage(
                timestamp=datetime.now().isoformat(),
                agent=agent,
                operation=operation,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                execution_time_ms=execution_time_ms,
                flow_name=flow_name
            )
            
            self.usage_records.append(usage)
            
            # Real-time logging
            print(f"📊 {agent}/{operation}: {total_tokens:,} tokens | ${cost:.4f} | {model}")
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate precise cost using Decimal for financial accuracy"""
        if model not in self.pricing:
            print(f"⚠️ Unknown model pricing: {model}, using default rates")
            pricing = {"input": 0.001, "output": 0.001}  # Default fallback
        else:
            pricing = self.pricing[model]
        
        # Use Decimal for precise financial calculations
        input_cost = Decimal(str(input_tokens)) * Decimal(str(pricing["input"])) / Decimal("1000")
        output_cost = Decimal(str(output_tokens)) * Decimal(str(pricing["output"])) / Decimal("1000")
        total_cost = input_cost + output_cost
        
        return float(total_cost.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
    
    def get_real_time_metrics(self) -> SessionMetrics:
        """Get current session metrics in real-time"""
        with self.lock:
            if not self.usage_records:
                return SessionMetrics(
                    session_start=self.session_start.isoformat(),
                    total_calls=0, total_tokens=0, total_cost=0.0,
                    flows_processed=0, average_tokens_per_flow=0.0,
                    average_cost_per_flow=0.0, current_tokens_per_minute=0.0,
                    estimated_daily_capacity=0, estimated_daily_cost=0.0
                )
            
            # Calculate basic metrics
            total_calls = len(self.usage_records)
            total_tokens = sum(r.total_tokens for r in self.usage_records)
            total_cost = sum(r.cost_usd for r in self.usage_records)
            
            # Flow-based calculations
            flows_processed = max(self.flow_count, 1)  # Avoid division by zero
            avg_tokens_per_flow = total_tokens / flows_processed
            avg_cost_per_flow = total_cost / flows_processed
            
            # Time-based calculations
            session_duration_minutes = (datetime.now() - self.session_start).total_seconds() / 60
            if session_duration_minutes > 0:
                tokens_per_minute = total_tokens / session_duration_minutes
            else:
                tokens_per_minute = 0
            
            # Daily capacity planning (8-hour work day)
            working_minutes_per_day = 8 * 60  # 480 minutes
            if avg_tokens_per_flow > 0 and tokens_per_minute > 0:
                daily_tokens = tokens_per_minute * working_minutes_per_day
                estimated_daily_capacity = int(daily_tokens / avg_tokens_per_flow)
                estimated_daily_cost = avg_cost_per_flow * estimated_daily_capacity
            else:
                estimated_daily_capacity = 0
                estimated_daily_cost = 0.0
            
            return SessionMetrics(
                session_start=self.session_start.isoformat(),
                total_calls=total_calls,
                total_tokens=total_tokens,
                total_cost=round(total_cost, 4),
                flows_processed=flows_processed,
                average_tokens_per_flow=round(avg_tokens_per_flow, 1),
                average_cost_per_flow=round(avg_cost_per_flow, 4),
                current_tokens_per_minute=round(tokens_per_minute, 1),
                estimated_daily_capacity=estimated_daily_capacity,
                estimated_daily_cost=round(estimated_daily_cost, 2)
            )
    
    def get_agent_breakdown(self) -> Dict[str, Dict]:
        """Get per-agent token and cost breakdown"""
        with self.lock:
            breakdown = {}
            for record in self.usage_records:
                agent = record.agent
                if agent not in breakdown:
                    breakdown[agent] = {
                        "calls": 0, "tokens": 0, "cost": 0.0,
                        "avg_tokens_per_call": 0, "efficiency_score": 0
                    }
                
                breakdown[agent]["calls"] += 1
                breakdown[agent]["tokens"] += record.total_tokens
                breakdown[agent]["cost"] += record.cost_usd
            
            # Calculate efficiency metrics
            for agent in breakdown:
                data = breakdown[agent]
                if data["calls"] > 0:
                    data["avg_tokens_per_call"] = round(data["tokens"] / data["calls"], 1)
                    # Efficiency: tokens per dollar (higher is better)
                    data["efficiency_score"] = round(data["tokens"] / max(data["cost"], 0.0001), 1)
                data["cost"] = round(data["cost"], 4)
            
            return breakdown
    
    def get_model_usage(self) -> Dict[str, Dict]:
        """Get per-model usage statistics"""
        with self.lock:
            model_stats = {}
            for record in self.usage_records:
                model = record.model
                if model not in model_stats:
                    model_stats[model] = {
                        "calls": 0, "tokens": 0, "cost": 0.0,
                        "avg_cost_per_1k_tokens": 0
                    }
                
                model_stats[model]["calls"] += 1
                model_stats[model]["tokens"] += record.total_tokens
                model_stats[model]["cost"] += record.cost_usd
            
            # Calculate cost efficiency
            for model in model_stats:
                data = model_stats[model]
                if data["tokens"] > 0:
                    data["avg_cost_per_1k_tokens"] = round((data["cost"] / data["tokens"]) * 1000, 4)
                data["cost"] = round(data["cost"], 4)
            
            return model_stats
    
    def mark_flow_complete(self, flow_name: str):
        """Mark a flow as completed for accurate capacity planning"""
        self.manual_track(
            agent="system",
            operation="flow_complete",
            model="none",
            input_tokens=0,
            output_tokens=0,
            flow_name=flow_name
        )
    
    def export_session_data(self, output_dir: str = "token_reports"):
        """Export session data to JSON and CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export detailed usage records
        json_file = os.path.join(output_dir, f"token_usage_{self.session_name}_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump({
                "session_info": {
                    "session_name": self.session_name,
                    "session_start": self.session_start.isoformat(),
                    "export_time": datetime.now().isoformat()
                },
                "metrics": asdict(self.get_real_time_metrics()),
                "agent_breakdown": self.get_agent_breakdown(),
                "model_usage": self.get_model_usage(),
                "detailed_records": [asdict(record) for record in self.usage_records]
            }, indent=2)
        
        # Export CSV summary
        csv_file = os.path.join(output_dir, f"token_summary_{self.session_name}_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Agent', 'Operation', 'Model', 'Tokens', 'Cost_USD', 'Flow'])
            for record in self.usage_records:
                writer.writerow([
                    record.timestamp, record.agent, record.operation,
                    record.model, record.total_tokens, record.cost_usd, record.flow_name or ''
                ])
        
        print(f"📄 Session data exported:")
        print(f"   JSON: {json_file}")
        print(f"   CSV: {csv_file}")
        
        return {"json": json_file, "csv": csv_file}
    
    def print_session_summary(self):
        """Print formatted session summary"""
        metrics = self.get_real_time_metrics()
        agent_breakdown = self.get_agent_breakdown()
        
        print("\n" + "="*60)
        print(f"📊 TOKEN USAGE SUMMARY - {self.session_name}")
        print("="*60)
        print(f"🕒 Session Duration: {(datetime.now() - self.session_start).total_seconds()/60:.1f} minutes")
        print(f"🔢 Total LLM Calls: {metrics.total_calls:,}")
        print(f"🎯 Total Tokens: {metrics.total_tokens:,}")
        print(f"💰 Total Cost: ${metrics.total_cost:.4f}")
        print(f"📈 Flows Processed: {metrics.flows_processed}")
        print(f"⚡ Avg Tokens/Flow: {metrics.average_tokens_per_flow:,.1f}")
        print(f"💵 Avg Cost/Flow: ${metrics.average_cost_per_flow:.4f}")
        print(f"⏱️  Tokens/Minute: {metrics.current_tokens_per_minute:,.1f}")
        print(f"📅 Est. Daily Capacity: {metrics.estimated_daily_capacity:,} flows")
        print(f"🏦 Est. Daily Cost: ${metrics.estimated_daily_cost:.2f}")
        
        print(f"\n🤖 AGENT PERFORMANCE:")
        for agent, stats in agent_breakdown.items():
            print(f"   {agent}: {stats['tokens']:,} tokens | ${stats['cost']:.4f} | {stats['calls']} calls")
        
        print("="*60)


    def calculate_capacity_planning(self, 
                               tokens_per_flow: int,
                               pending_flows: int, 
                               daily_hours: float,
                               minutes_per_flow: int) -> Dict[str, Any]:
        """
        Calculate capacity planning metrics for ACE flow conversion
        
        Args:
            tokens_per_flow: Number of tokens consumed per flow
            pending_flows: Total number of flows to be converted
            daily_hours: Hours available per day for processing
            minutes_per_flow: Time in minutes required per flow
            
        Returns:
            Dict containing all capacity planning metrics and costs
        """
        # Basic capacity calculations
        daily_minutes = daily_hours * 60
        flows_per_day = int(daily_minutes / minutes_per_flow)
        flows_per_month = flows_per_day * 20  # 20 working days per month
        
        # Token usage calculations
        daily_tokens = flows_per_day * tokens_per_flow
        monthly_tokens = daily_tokens * 20
        
        # Time to complete all pending flows
        days_to_complete = pending_flows / flows_per_day if flows_per_day > 0 else 0
        
        # Cost calculations for both models
        llama_8b_daily_cost = (daily_tokens / 1000) * 0.000050
        llama_8b_monthly_cost = llama_8b_daily_cost * 20
        llama_8b_cost_per_flow = llama_8b_monthly_cost / flows_per_month if flows_per_month > 0 else 0
        
        llama_70b_daily_cost = (daily_tokens / 1000) * 0.000800
        llama_70b_monthly_cost = llama_70b_daily_cost * 20
        llama_70b_cost_per_flow = llama_70b_monthly_cost / flows_per_month if flows_per_month > 0 else 0
        
        # Determine required subscription tier
        if daily_tokens <= 100000:
            required_tier = "Free Tier"
            tier_cost = 0
        elif daily_tokens <= 1000000:
            required_tier = "Dev Tier"
            tier_cost = 20
        else:
            required_tier = "Pro Tier"
            tier_cost = 100
        
        # Calculate completion timeline
        months_to_complete = days_to_complete / 20 if days_to_complete > 20 else days_to_complete / 20
        
        return {
            "capacity_metrics": {
                "flows_per_day": flows_per_day,
                "flows_per_month": flows_per_month,
                "daily_tokens": daily_tokens,
                "monthly_tokens": monthly_tokens,
                "days_to_complete_all": round(days_to_complete, 1),
                "months_to_complete_all": round(months_to_complete, 1)
            },
            "cost_analysis": {
                "llama_8b": {
                    "daily_cost": round(llama_8b_daily_cost, 4),
                    "monthly_cost": round(llama_8b_monthly_cost, 2),
                    "cost_per_flow": round(llama_8b_cost_per_flow, 4),
                    "total_cost_for_all_flows": round((pending_flows * tokens_per_flow / 1000) * 0.000050, 2)
                },
                "llama_70b": {
                    "daily_cost": round(llama_70b_daily_cost, 4),
                    "monthly_cost": round(llama_70b_monthly_cost, 2),
                    "cost_per_flow": round(llama_70b_cost_per_flow, 4),
                    "total_cost_for_all_flows": round((pending_flows * tokens_per_flow / 1000) * 0.000800, 2)
                }
            },
            "subscription_requirements": {
                "required_tier": required_tier,
                "tier_monthly_cost": tier_cost,
                "daily_token_limit_needed": daily_tokens,
                "exceeds_free_tier": daily_tokens > 100000,
                "exceeds_dev_tier": daily_tokens > 1000000
            },
            "efficiency_metrics": {
                "utilization_rate": round((flows_per_day * minutes_per_flow) / daily_minutes * 100, 1),
                "tokens_per_minute": round(daily_tokens / daily_minutes, 0),
                "cost_difference_8b_vs_70b": round(llama_70b_monthly_cost - llama_8b_monthly_cost, 2),
                "savings_using_8b": round(((llama_70b_monthly_cost - llama_8b_monthly_cost) / llama_70b_monthly_cost) * 100, 1)
            },
            "input_parameters": {
                "tokens_per_flow": tokens_per_flow,
                "pending_flows": pending_flows,
                "daily_hours": daily_hours,
                "minutes_per_flow": minutes_per_flow
            }
        }    


# Convenience functions for easy integration
def create_tracker(session_name: str = None) -> SmartTokenTracker:
    """Create a new token tracker instance"""
    return SmartTokenTracker(session_name)


def wrap_groq_client(client, tracker: SmartTokenTracker, agent: str):
    """Wrap a Groq client to automatically track all calls"""
    original_create = client.chat.completions.create
    
    def tracked_create(*args, **kwargs):
        # Extract operation from messages or use default
        operation = "llm_call"
        if 'messages' in kwargs and kwargs['messages']:
            content = str(kwargs['messages'])
            if 'esql' in content.lower():
                operation = "esql_generation"
            elif 'schema' in content.lower():
                operation = "schema_generation"
            elif 'transform' in content.lower():
                operation = "transformation"
        
        @tracker.track_llm_call(agent, operation)
        def make_call():
            return original_create(*args, **kwargs)
        
        return make_call()
    
    client.chat.completions.create = tracked_create
    return client


# Example integration helper
class TrackedGroqClient:
    """Groq client wrapper with automatic token tracking"""
    
    def __init__(self, groq_client, tracker: SmartTokenTracker, agent_name: str):
        self.client = groq_client
        self.tracker = tracker
        self.agent_name = agent_name
    
    def chat_completion(self, operation: str, flow_name: str = None, **kwargs):
        """Tracked chat completion call"""
        @self.tracker.track_llm_call(self.agent_name, operation, flow_name)
        def make_call():
            return self.client.chat.completions.create(**kwargs)
        
        return make_call()


if __name__ == "__main__":
    # Example usage demonstration
    tracker = create_tracker("demo_session")
    
    # Simulate some LLM calls
    tracker.manual_track("agent_1", "esql_generation", "llama-3.3-70b-versatile", 1500, 800, 2000, "flow_1")
    tracker.manual_track("agent_2", "schema_analysis", "llama-3.1-8b-instant", 800, 400, 1200, "flow_1")
    tracker.mark_flow_complete("flow_1")
    
    # Show real-time metrics
    metrics = tracker.get_real_time_metrics()
    print(f"Current session cost: ${metrics.total_cost}")
    print(f"Estimated daily capacity: {metrics.estimated_daily_capacity} flows")
    
    # Print summary
    tracker.print_session_summary()
    
    # Export data
    tracker.export_session_data()
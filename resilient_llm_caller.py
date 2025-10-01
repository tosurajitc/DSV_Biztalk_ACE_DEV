"""
Resilient LLM Caller with Exponential Backoff and Rate Limiting
Handles timeouts, rate limits, and partial failures gracefully.
"""

import time
import logging
from typing import Optional, Dict, Any
from groq import Groq
import os

logger = logging.getLogger(__name__)


class ResilientLLMCaller:
    """
    Wrapper for LLM API calls with built-in resilience.
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "llama-3.3-70b-versatile",
        default_timeout: int = 180,
        rate_limit_delay: float = 3.0
    ):
        """
        Initialize resilient LLM caller.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model name to use
            default_timeout: Default timeout per request in seconds
            rate_limit_delay: Delay between consecutive calls in seconds
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.default_timeout = default_timeout
        self.rate_limit_delay = rate_limit_delay
        self.last_call_time = 0
        
        # Initialize Groq client with timeout
        self.client = Groq(
            api_key=self.api_key,
            timeout=default_timeout
        )
    
    def _enforce_rate_limit(self):
        """Ensure minimum delay between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.info(f"⏳ Rate limiting: sleeping {sleep_time:.2f}s...")
            time.sleep(sleep_time)
    
    def call_with_retry(
        self,
        prompt: str,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        timeout: int = None,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        Call LLM with exponential backoff retry logic.
        
        Args:
            prompt: The prompt to send
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (doubles each time)
            max_delay: Maximum delay between retries
            timeout: Override default timeout for this call
            temperature: LLM temperature setting
            
        Returns:
            LLM response text or None if all retries failed
        """
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        timeout = timeout or self.default_timeout
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 LLM call attempt {attempt + 1}/{max_retries}...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert IBM ACE developer assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    timeout=timeout
                )
                
                # Success! Update last call time and return
                self.last_call_time = time.time()
                
                result = response.choices[0].message.content
                logger.info(f"✅ LLM call succeeded ({len(result)} chars)")
                
                return result
                
            except TimeoutError as e:
                logger.warning(f"⏱️ Timeout on attempt {attempt + 1}/{max_retries}")
                
                if attempt == max_retries - 1:
                    logger.error(f"❌ Failed after {max_retries} timeout attempts")
                    return None
                
                # Exponential backoff: 2s, 4s, 8s, 16s, 32s (capped at max_delay)
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.info(f"⏳ Retrying in {delay}s...")
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"❌ Unexpected error on attempt {attempt + 1}/{max_retries}: {str(e)}")
                
                if attempt == max_retries - 1:
                    logger.error(f"❌ Failed after {max_retries} attempts")
                    return None
                
                # Still retry on unexpected errors
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.info(f"⏳ Retrying in {delay}s...")
                time.sleep(delay)
        
        return None
    
    def call_batch_with_partial_success(
        self,
        items: list,
        prompt_generator_fn,
        processor_fn,
        checkpoint_file: str = None
    ) -> Dict[str, Any]:
        """
        Process multiple items with LLM, tracking partial success.
        
        Args:
            items: List of items to process
            prompt_generator_fn: Function that takes an item and returns a prompt
            processor_fn: Function that takes (item, llm_response) and processes it
            checkpoint_file: Optional file to save progress
            
        Returns:
            Dictionary with 'successful', 'failed', and 'results' keys
        """
        results = {
            'successful': [],
            'failed': [],
            'results': {}
        }
        
        # Load checkpoint if exists
        if checkpoint_file and os.path.exists(checkpoint_file):
            import json
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                completed = set(checkpoint.get('completed', []))
                logger.info(f"📂 Loaded checkpoint: {len(completed)} items already completed")
                items = [item for item in items if item not in completed]
        
        total = len(items)
        
        for idx, item in enumerate(items, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔨 [{idx}/{total}] Processing: {item}")
            logger.info(f"{'='*60}")
            
            try:
                # Generate prompt
                prompt = prompt_generator_fn(item)
                
                # Call LLM with retry
                response = self.call_with_retry(prompt)
                
                if response:
                    # Process response
                    result = processor_fn(item, response)
                    
                    if result:
                        results['successful'].append(item)
                        results['results'][item] = result
                        logger.info(f"✅ [{idx}/{total}] {item} completed successfully")
                    else:
                        results['failed'].append(item)
                        logger.error(f"❌ [{idx}/{total}] {item} processing failed")
                else:
                    results['failed'].append(item)
                    logger.error(f"❌ [{idx}/{total}] {item} LLM call failed")
                
                # Save checkpoint
                if checkpoint_file:
                    import json
                    checkpoint_data = {
                        'completed': results['successful'],
                        'failed': results['failed'],
                        'timestamp': time.time()
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=2)
                
            except Exception as e:
                logger.error(f"❌ [{idx}/{total}] Exception for {item}: {str(e)}")
                results['failed'].append(item)
                continue
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Batch Processing Summary:")
        logger.info(f"  ✅ Successful: {len(results['successful'])}/{total}")
        logger.info(f"  ❌ Failed: {len(results['failed'])}/{total}")
        logger.info(f"{'='*60}")
        
        if results['failed']:
            logger.warning(f"\n🔄 Failed items (can retry manually):")
            for item in results['failed']:
                logger.warning(f"    - {item}")
        
        return results


# Global singleton instance
_llm_caller = None

def get_llm_caller(**kwargs) -> ResilientLLMCaller:
    """Get or create global LLM caller instance."""
    global _llm_caller
    if _llm_caller is None:
        _llm_caller = ResilientLLMCaller(**kwargs)
    return _llm_caller
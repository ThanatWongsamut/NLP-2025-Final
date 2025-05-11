#!/usr/bin/env python3
"""
Script to prepare QMSum dataset for fine-tuning Gemma-3-27b-it
Focuses on processing the specific_query_list portion of the dataset
"""

import os
import json
import logging
from tqdm import tqdm
from datasets import Dataset

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def prepare_qmsum_dataset(data_dir="QMSum/data/ALL", split="train"):
    """
    Process the QMSum dataset for fine-tuning, focusing on specific queries
    
    Args:
        data_dir: Path to QMSum data directory
        split: Data split to process (train, val, test)
        
    Returns:
        Dataset object with processed examples
    """
    logger.info(f"Processing {split} split from {data_dir}")
    
    # Check if directory exists
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        logger.error(f"Directory not found: {split_dir}")
        raise FileNotFoundError(f"Directory not found: {split_dir}")
    
    examples = []

    files = [f for f in os.listdir(split_dir) if f.endswith('.json')]
    
    if not files:
        logger.warning(f"No JSON files found in {split_dir}")
        return Dataset.from_dict({"prompt": [], "completion": []})
    
    # Process each JSON file
    for filename in tqdm(files, desc=f"Processing {split} files"):
        file_path = os.path.join(split_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract the meeting transcripts
            meeting_transcripts = data.get("meeting_transcripts", [])
            
            # Focus on specific_query_list as requested
            specific_queries = data.get("specific_query_list", [])
            
            for item in specific_queries:
                query = item.get("query", "")
                answer = item.get("answer", "")
                relevant_spans = item.get("relevant_text_span", [])
                
                # Skip if any required field is missing
                if not query or not answer or not relevant_spans:
                    continue
                
                # Extract relevant transcript segments
                relevant_transcripts = []
                for span in relevant_spans:
                    if len(span) != 2:
                        continue
                    
                    try:
                        start_idx, end_idx = int(span[0]), int(span[1])
                        # Check if indices are valid
                        if 0 <= start_idx <= end_idx < len(meeting_transcripts):
                            relevant_transcripts.extend(meeting_transcripts[start_idx:end_idx+1])
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error processing span {span} in {filename}: {e}")
                        continue
                
                # Create transcript text
                transcript_text = ""
                for segment in relevant_transcripts:
                    speaker = segment.get('speaker', 'Unknown Speaker')
                    content = segment.get('content', '')
                    transcript_text += f"{speaker}: {content}\n"
                
                # Skip if transcript is empty
                if not transcript_text.strip():
                    continue
                
                # Create the prompt
                prompt = f"""You are an assistant that answers questions about meeting transcripts.

Meeting Transcript:
{transcript_text}

Question: {query}

Answer the question based ONLY on the information provided in the transcript.
Be concise and factual. If the information is not in the transcript, say "The transcript does not provide this information."
"""
                examples.append({
                    "prompt": prompt,
                    "completion": answer,
                })
        
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            continue
    
    logger.info(f"Processed {len(examples)} examples from {split} split")
    
    # Create dataset
    return Dataset.from_dict({
        "prompt": [ex["prompt"] for ex in examples],
        "completion": [ex["completion"] for ex in examples]
    })

def analyze_dataset(dataset):
    """Print basic statistics about the dataset"""
    prompt_lengths = [len(p.split()) for p in dataset["prompt"]]
    completion_lengths = [len(c.split()) for c in dataset["completion"]]
    
    logger.info(f"Dataset size: {len(dataset)} examples")
    logger.info(f"Average prompt length: {sum(prompt_lengths)/len(prompt_lengths):.1f} words")
    logger.info(f"Average completion length: {sum(completion_lengths)/len(completion_lengths):.1f} words")
    logger.info(f"Max prompt length: {max(prompt_lengths)} words")
    logger.info(f"Max completion length: {max(completion_lengths)} words")

def main():
    # Create output directory
    os.makedirs("processed_data", exist_ok=True)
    
    # Process each split
    for split in ["train", "val", "test"]:
        try:
            dataset = prepare_qmsum_dataset(split=split)
            analyze_dataset(dataset)
            
            # Save the processed dataset
            output_dir = os.path.join("processed_data", split)
            dataset.save_to_disk(output_dir)
            logger.info(f"Saved {split} dataset to {output_dir}")
        
        except Exception as e:
            logger.error(f"Error processing {split} split: {e}")

if __name__ == "__main__":
    main()
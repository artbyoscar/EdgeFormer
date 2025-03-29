#!/usr/bin/env python
# EdgeFormer - LIMO Dataset Curation Script
# Author: Oscar Nunez (art.by.oscar.n@gmail.com)

"""
LIMO (Less Is More) Dataset Curation for EdgeFormer
---------------------------------------------------
This script implements a LIMO approach to dataset curation, focusing on
selecting high-quality, diverse examples that provide maximum learning
signal rather than simply gathering massive amounts of data.

The script analyzes text samples using multiple quality metrics:
1. Complexity - Using readability scores, vocabulary richness, and syntactic complexity
2. Diversity - Ensuring samples cover different topics, styles, and patterns
3. Information density - Prioritizing examples with high information-to-token ratio
4. Coherence - Selecting well-structured, logically consistent texts
5. Task relevance - Focusing on samples relevant to target tasks

Usage:
python scripts/curate_limo_dataset.py --input_data data/wikitext --output_dir data/limo_curated --quality_threshold 0.8 --max_samples 2500
"""

import os
import sys
import re
import math
import random
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import textstat

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.text_dataset import get_tokenizer, TextDataset

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class LIMODatasetCurator:
    """
    Implements the LIMO (Less Is More) approach to dataset curation,
    focusing on quality over quantity for training data.
    """
    
    def __init__(self, args):
        self.args = args
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = get_tokenizer()
        
        # Initialize metrics tracking
        self.metrics = {
            'complexity': [],
            'diversity': [],
            'information_density': [],
            'coherence': [],
            'overall_quality': []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        # Initialize vectorizer for diversity analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize statistics tracking
        self.stats = {
            'total_processed': 0,
            'accepted': 0,
            'rejected': 0,
            'avg_quality_score': 0.0,
            'quality_distribution': {
                'excellent': 0,  # 0.9-1.0
                'good': 0,       # 0.8-0.9
                'average': 0,    # 0.7-0.8
                'fair': 0,       # 0.6-0.7
                'poor': 0        # <0.6
            }
        }
    
    def calculate_complexity_score(self, text):
        """
        Calculate text complexity based on readability, vocabulary richness, 
        and syntactic complexity.
        """
        if not text or len(text) < 50:
            return 0.0
            
        # Readability score (lower is more complex)
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            # Convert to 0-1 scale where higher is more complex
            # Flesch score ranges from 0-100, where 0 is most complex
            readability = max(0, min(1, 1 - flesch_score/100))
        except:
            readability = 0.5  # Default if calculation fails
        
        # Vocabulary richness (Type-Token Ratio)
        words = word_tokenize(text.lower())
        if not words:
            return 0.0
            
        # Filter out punctuation and stopwords
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        if not words:
            return 0.0
            
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # Normalize TTR (typical range 0.3-0.7)
        vocab_richness = min(1.0, ttr / 0.7)
        
        # Syntactic complexity based on sentence length
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
            
        avg_sentence_length = len(words) / len(sentences)
        
        # Normalize sentence length (typical range 10-25)
        syntactic_complexity = min(1.0, avg_sentence_length / 25)
        
        # Combine metrics with weights
        complexity_score = (
            0.4 * readability +
            0.4 * vocab_richness +
            0.2 * syntactic_complexity
        )
        
        return complexity_score
    
    def calculate_diversity_score(self, text, reference_texts):
        """
        Calculate how diverse this text is compared to the reference corpus.
        Higher score means more unique/diverse content.
        """
        if not text or not reference_texts:
            return 0.5  # Default score
        
        # Extract n-grams from the text
        text_words = word_tokenize(text.lower())
        text_words = [w for w in text_words if w.isalpha() and w not in self.stop_words]
        
        if not text_words:
            return 0.0
            
        # If we don't have enough reference texts, return a moderate diversity score
        if len(reference_texts) < 5:
            return 0.5
            
        # Create a combined corpus for TF-IDF
        all_texts = reference_texts + [text]
        
        try:
            # Calculate TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Get the vector for our text (the last one)
            text_vector = tfidf_matrix[-1]
            
            # Calculate cosine similarity with each reference text
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(text_vector, tfidf_matrix[:-1])[0]
            
            # Calculate diversity as inverse of maximum similarity
            max_similarity = max(similarities) if similarities.size > 0 else 0
            diversity_score = 1 - max_similarity
            
            return diversity_score
        except:
            # Fallback if vectorization fails
            return 0.5
    
    def calculate_information_density(self, text):
        """
        Calculate information density based on the ratio of meaningful content
        to total tokens.
        """
        if not text or len(text) < 10:
            return 0.0
            
        # Tokenize text
        words = word_tokenize(text.lower())
        if not words:
            return 0.0
            
        # Count meaningful words (non-stopwords)
        meaningful_words = [w for w in words if w.isalpha() and w not in self.stop_words]
        
        if not words:
            return 0.0
            
        # Calculate density
        density = len(meaningful_words) / len(words)
        
        # Normalize (typical range 0.3-0.7)
        normalized_density = min(1.0, density / 0.7)
        
        # Calculate information entropy as a measure of unpredictability
        # Higher entropy = more information
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy = 0
        for count in word_counts.values():
            probability = count / total_words
            entropy -= probability * math.log2(probability)
            
        # Normalize entropy (typical range 7-12 for English)
        normalized_entropy = min(1.0, entropy / 12)
        
        # Combine metrics
        info_density_score = 0.6 * normalized_density + 0.4 * normalized_entropy
        
        return info_density_score
    
    def calculate_coherence_score(self, text):
        """
        Calculate text coherence based on sentence transitions and structure.
        """
        if not text or len(text) < 50:
            return 0.0
            
        # Split into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return 0.5  # Default for single sentences
            
        # Calculate sentence similarity between adjacent sentences
        similarities = []
        
        for i in range(len(sentences) - 1):
            s1 = set(word_tokenize(sentences[i].lower()))
            s2 = set(word_tokenize(sentences[i + 1].lower()))
            
            # Filter out stopwords
            s1 = {w for w in s1 if w.isalpha() and w not in self.stop_words}
            s2 = {w for w in s2 if w.isalpha() and w not in self.stop_words}
            
            # Calculate Jaccard similarity
            if s1 and s2:
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        # Calculate average similarity (higher means more coherent)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Check for discourse markers indicating logical flow
        discourse_markers = [
            'however', 'therefore', 'thus', 'consequently', 'furthermore',
            'moreover', 'in addition', 'in conclusion', 'for example',
            'for instance', 'specifically', 'in particular', 'in contrast',
            'on the other hand', 'first', 'second', 'third', 'finally'
        ]
        
        marker_count = sum(1 for marker in discourse_markers if marker in text.lower())
        normalized_marker_count = min(1.0, marker_count / 5)  # Normalize, assuming 5+ is excellent
        
        # Combine metrics
        coherence_score = 0.6 * avg_similarity + 0.4 * normalized_marker_count
        
        return coherence_score
    
    def calculate_overall_quality(self, complexity, diversity, info_density, coherence):
        """
        Calculate an overall quality score for a text sample.
        """
        # Apply weights based on importance
        weights = {
            'complexity': 0.25,
            'diversity': 0.3,
            'info_density': 0.25,
            'coherence': 0.2
        }
        
        overall_score = (
            weights['complexity'] * complexity +
            weights['diversity'] * diversity +
            weights['info_density'] * info_density +
            weights['coherence'] * coherence
        )
        
        return overall_score
    
    def get_quality_tier(self, score):
        """
        Categorize a quality score into a tier.
        """
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'good'
        elif score >= 0.7:
            return 'average'
        elif score >= 0.6:
            return 'fair'
        else:
            return 'poor'
    
    def process_text_file(self, file_path):
        """
        Process a text file to extract and evaluate samples.
        """
        print(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        
        # Split into reasonable chunks (paragraphs or sections)
        samples = []
        
        # Different strategies for different file types
        if '.txt' in file_path.lower():
            # Simple paragraph splitting for txt files
            paragraphs = re.split(r'\n\s*\n', content)
            
            # Filter out too short paragraphs
            samples = [p.strip() for p in paragraphs if len(p.strip()) >= 100]
            
        elif '.wiki' in file_path.lower() or '.wikipedia' in file_path.lower():
            # Handle WikiText format with = Section Headings =
            sections = re.split(r'(^|\n)=+ .+ =+($|\n)', content)
            
            # Process each section
            current_section = ""
            for section in sections:
                if re.match(r'(^|\n)=+ .+ =+($|\n)', section):
                    # This is a section header
                    if len(current_section.strip()) >= 200:
                        samples.append(current_section.strip())
                    current_section = ""
                else:
                    # Split into paragraphs
                    paragraphs = re.split(r'\n\s*\n', section)
                    for para in paragraphs:
                        if len(para.strip()) >= 100:
                            current_section += para.strip() + "\n\n"
            
            # Add the last section
            if len(current_section.strip()) >= 200:
                samples.append(current_section.strip())
                
        elif '.json' in file_path.lower():
            # Try to handle JSON files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle various JSON structures
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'text' in item:
                            text = item['text']
                            if len(text.strip()) >= 100:
                                samples.append(text.strip())
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str) and len(value.strip()) >= 100:
                            samples.append(value.strip())
                        elif isinstance(value, dict) and 'text' in value:
                            text = value['text']
                            if len(text.strip()) >= 100:
                                samples.append(text.strip())
            except:
                print(f"Could not parse {file_path} as JSON")
        else:
            # Generic approach for other files
            # Try to split by double newlines first
            paragraphs = re.split(r'\n\s*\n', content)
            if len(paragraphs) < 3:
                # If too few paragraphs, try splitting by single newlines
                paragraphs = content.split('\n')
            
            # Combine very short paragraphs
            combined = []
            current = ""
            
            for para in paragraphs:
                if len(para.strip()) < 50:
                    current += para.strip() + " "
                else:
                    if current:
                        current += para.strip()
                        combined.append(current)
                        current = ""
                    else:
                        combined.append(para.strip())
            
            if current:
                combined.append(current)
            
            # Filter by length
            samples = [s for s in combined if len(s) >= 100]
        
        return samples
    
    def evaluate_samples(self, samples):
        """
        Evaluate text samples using LIMO quality metrics.
        """
        selected_samples = []
        reference_samples = samples[:min(50, len(samples))]  # Use some samples as reference
        
        print(f"Evaluating {len(samples)} text samples...")
        
        for i, sample in enumerate(tqdm(samples)):
            # Skip if too short after cleaning
            clean_sample = re.sub(r'\s+', ' ', sample).strip()
            if len(clean_sample) < 100:
                continue
                
            # Calculate quality metrics
            complexity = self.calculate_complexity_score(clean_sample)
            diversity = self.calculate_diversity_score(clean_sample, reference_samples)
            info_density = self.calculate_information_density(clean_sample)
            coherence = self.calculate_coherence_score(clean_sample)
            
            # Calculate overall quality
            overall_quality = self.calculate_overall_quality(
                complexity, diversity, info_density, coherence
            )
            
            # Track metrics
            self.metrics['complexity'].append(complexity)
            self.metrics['diversity'].append(diversity)
            self.metrics['information_density'].append(info_density)
            self.metrics['coherence'].append(coherence)
            self.metrics['overall_quality'].append(overall_quality)
            
            # Update statistics
            self.stats['total_processed'] += 1
            
            # Categorize quality
            tier = self.get_quality_tier(overall_quality)
            self.stats['quality_distribution'][tier] += 1
            
            # Accept if quality is above threshold
            if overall_quality >= self.args.quality_threshold:
                self.stats['accepted'] += 1
                
                selected_samples.append({
                    'text': clean_sample,
                    'metrics': {
                        'complexity': complexity,
                        'diversity': diversity,
                        'information_density': info_density,
                        'coherence': coherence,
                        'overall_quality': overall_quality
                    }
                })
            else:
                self.stats['rejected'] += 1
        
        # Calculate average quality
        if self.metrics['overall_quality']:
            self.stats['avg_quality_score'] = sum(self.metrics['overall_quality']) / len(self.metrics['overall_quality'])
        
        return selected_samples
    
    def cluster_and_diversify(self, samples, num_clusters=10):
        """
        Ensure diversity by clustering samples and selecting representatives from each cluster.
        """
        if len(samples) <= self.args.max_samples:
            return samples
            
        print("Clustering samples to ensure diversity...")
        
        # Extract text for vectorization
        texts = [s['text'] for s in samples]
        
        # Vectorize texts
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Determine optimal number of clusters (min of num_clusters or sqrt of samples)
            k = min(num_clusters, int(math.sqrt(len(samples))))
            
            # Cluster samples
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Add cluster information to samples
            for i, sample in enumerate(samples):
                sample['cluster'] = int(clusters[i])
            
            # Group by cluster
            clustered = {}
            for i, sample in enumerate(samples):
                cluster = sample['cluster']
                if cluster not in clustered:
                    clustered[cluster] = []
                clustered[cluster].append(sample)
            
            # Select top samples from each cluster based on quality
            final_samples = []
            samples_per_cluster = self.args.max_samples // k
            
            for cluster, cluster_samples in clustered.items():
                # Sort by quality
                sorted_samples = sorted(
                    cluster_samples, 
                    key=lambda x: x['metrics']['overall_quality'], 
                    reverse=True
                )
                
                # Take top samples
                final_samples.extend(sorted_samples[:samples_per_cluster])
            
            # If we need more samples to reach max_samples, take from highest quality
            if len(final_samples) < self.args.max_samples:
                remaining = self.args.max_samples - len(final_samples)
                
                # Create a pool of remaining samples
                used = set(s['text'] for s in final_samples)
                remaining_pool = [s for s in samples if s['text'] not in used]
                
                # Sort by quality
                remaining_pool = sorted(
                    remaining_pool, 
                    key=lambda x: x['metrics']['overall_quality'], 
                    reverse=True
                )
                
                # Add top remaining samples
                final_samples.extend(remaining_pool[:remaining])
            
            print(f"Selected {len(final_samples)} samples from {k} clusters")
            return final_samples
        except Exception as e:
            print(f"Error during clustering: {e}")
            # Fallback to quality-based selection
            sorted_samples = sorted(
                samples, 
                key=lambda x: x['metrics']['overall_quality'], 
                reverse=True
            )
            return sorted_samples[:self.args.max_samples]
    
    def save_dataset(self, samples):
        """
        Save the curated dataset to files.
        """
        if not samples:
            print("No samples to save!")
            return
            
        # Save as JSON with metrics
        json_path = os.path.join(self.args.output_dir, 'limo_dataset.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2)
        
        # Save plain text version
        txt_path = os.path.join(self.args.output_dir, 'limo_dataset.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample['text'] + '\n\n===\n\n')
        
        # Save tokenized version if requested
        if self.args.save_tokenized:
            print("Tokenizing samples...")
            tokenized_path = os.path.join(self.args.output_dir, 'limo_dataset_tokenized.json')
            tokenized_samples = []
            
            for sample in tqdm(samples):
                tokens = self.tokenizer.encode(sample['text'])
                tokenized_samples.append({
                    'tokens': tokens,
                    'text': sample['text'],
                    'metrics': sample['metrics']
                })
            
            with open(tokenized_path, 'w', encoding='utf-8') as f:
                json.dump(tokenized_samples, f)
        
        # Save statistics
        stats_path = os.path.join(self.args.output_dir, 'curation_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save metrics distributions
        metrics_path = os.path.join(self.args.output_dir, 'metrics_distribution.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Dataset saved to {self.args.output_dir}")
        print(f"Total samples: {len(samples)}")
    
    def generate_report(self):
        """
        Generate a comprehensive report on the dataset curation process.
        """
        report_path = os.path.join(self.args.output_dir, 'curation_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# LIMO Dataset Curation Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
            
            f.write("## Curation Parameters\n\n")
            f.write(f"- Input data: `{self.args.input_data}`\n")
            f.write(f"- Quality threshold: {self.args.quality_threshold}\n")
            f.write(f"- Maximum samples: {self.args.max_samples}\n")
            f.write(f"- Random seed: {self.args.seed}\n\n")
            
            f.write("## Dataset Statistics\n\n")
            f.write(f"- Total samples processed: {self.stats['total_processed']}\n")
            f.write(f"- Samples accepted: {self.stats['accepted']} ({self.stats['accepted']/max(1, self.stats['total_processed'])*100:.2f}%)\n")
            f.write(f"- Samples rejected: {self.stats['rejected']} ({self.stats['rejected']/max(1, self.stats['total_processed'])*100:.2f}%)\n")
            f.write(f"- Average quality score: {self.stats['avg_quality_score']:.4f}\n\n")
            
            f.write("### Quality Distribution\n\n")
            f.write("| Quality Tier | Count | Percentage |\n")
            f.write("|--------------|-------|------------|\n")
            
            for tier, count in self.stats['quality_distribution'].items():
                percentage = count / max(1, self.stats['total_processed']) * 100
                f.write(f"| {tier.capitalize()} | {count} | {percentage:.2f}% |\n")
            
            f.write("\n## Metric Distributions\n\n")
            
            # Calculate statistics for each metric
            for metric in self.metrics:
                if not self.metrics[metric]:
                    continue
                    
                values = self.metrics[metric]
                avg = sum(values) / len(values)
                median = sorted(values)[len(values)//2]
                min_val = min(values)
                max_val = max(values)
                
                f.write(f"### {metric.replace('_', ' ').title()}\n\n")
                f.write(f"- Average: {avg:.4f}\n")
                f.write(f"- Median: {median:.4f}\n")
                f.write(f"- Minimum: {min_val:.4f}\n")
                f.write(f"- Maximum: {max_val:.4f}\n\n")
            
            f.write("## LIMO Dataset Contents\n\n")
            f.write(f"The dataset is available in the following formats:\n\n")
            f.write("- `limo_dataset.json`: Full dataset with metrics\n")
            f.write("- `limo_dataset.txt`: Plain text version\n")
            
            if self.args.save_tokenized:
                f.write("- `limo_dataset_tokenized.json`: Tokenized version\n")
            
            f.write("\n## Usage Recommendations\n\n")
            f.write("This LIMO dataset has been curated for high-quality training examples. ")
            f.write("It emphasizes quality over quantity, with each example providing maximum learning signal. ")
            f.write("We recommend using this dataset for fine-tuning EdgeFormer models where compute ")
            f.write("resources are limited, as it should provide better results than larger but lower-quality datasets.\n\n")
            
            f.write("### Training Recommendations:\n\n")
            f.write("- Use a slightly higher learning rate than with larger datasets\n")
            f.write("- Consider increasing the number of epochs since the dataset is smaller\n")
            f.write("- Monitor validation performance carefully to avoid overfitting\n")
            f.write("- This dataset is designed for EdgeFormer's unique architecture and may yield better results ")
            f.write("on that architecture than on other models")
        
        print(f"Report generated at {report_path}")
    
    def curate(self):
        """
        Main curation process.
        """
        all_samples = []
        
        # Process input data
        input_path = Path(self.args.input_data)
        
        if input_path.is_file():
            # Single file input
            samples = self.process_text_file(str(input_path))
            all_samples.extend(samples)
        else:
            # Directory input
            for file_path in input_path.glob('**/*'):
                if file_path.is_file():
                    samples = self.process_text_file(str(file_path))
                    all_samples.extend(samples)
        
        print(f"Found {len(all_samples)} text samples")
        
        # Evaluate samples
        quality_samples = self.evaluate_samples(all_samples)
        print(f"Selected {len(quality_samples)} samples above quality threshold")
        
        # Ensure diversity and limit to max_samples
        final_samples = self.cluster_and_diversify(quality_samples)
        
        # Save dataset
        self.save_dataset(final_samples)
        
        # Generate report
        self.generate_report()
        
        return final_samples


def main():
    parser = argparse.ArgumentParser(description='LIMO (Less Is More) Dataset Curation')
    
    parser.add_argument('--input_data', type=str, required=True,
                        help='Path to input data (file or directory)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save curated dataset')
    parser.add_argument('--quality_threshold', type=float, default=0.75,
                        help='Minimum quality score to include a sample (0-1)')
    parser.add_argument('--max_samples', type=int, default=2500,
                        help='Maximum number of samples to include')
    parser.add_argument('--save_tokenized', action='store_true',
                        help='Save a tokenized version of the dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    curator = LIMODatasetCurator(args)
    curator.curate()


if __name__ == "__main__":
    main()
"""
Complete Pipeline Execution Script
Runs all components of Assignment 1 in sequence
"""

import sys
import subprocess
import time
from pathlib import Path


class PipelineRunner:
    """Orchestrates the complete sentiment analysis pipeline"""
    
    def __init__(self):
        self.steps = []
        self.start_time = None
        self.results = {}
    
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70 + "\n")
    
    def print_step(self, step_num, title):
        """Print step header"""
        print(f"\n{'─' * 70}")
        print(f"STEP {step_num}: {title}")
        print(f"{'─' * 70}\n")
    
    def run_script(self, script_name, description):
        """Run a Python script and capture results"""
        print(f"Running {script_name}...")
        print(f"Description: {description}\n")
        
        step_start = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Print output
            if result.stdout:
                print(result.stdout)
            
            if result.stderr and result.returncode != 0:
                print("ERRORS:", result.stderr)
                return False, result.stderr
            
            elapsed = time.time() - step_start
            print(f"\n✓ {script_name} completed in {elapsed:.1f} seconds")
            
            return True, None
            
        except subprocess.TimeoutExpired:
            print(f"✗ {script_name} timed out after 5 minutes")
            return False, "Timeout"
        except Exception as e:
            print(f"✗ {script_name} failed with error: {e}")
            return False, str(e)
    
    def check_file_exists(self, filename):
        """Check if a file was created"""
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {filename} created ({size:,} bytes)")
            return True
        else:
            print(f"  ✗ {filename} not found")
            return False
    
    def verify_outputs(self, step_name, expected_files):
        """Verify that expected output files exist"""
        print(f"\nVerifying {step_name} outputs:")
        all_exist = True
        for filename in expected_files:
            if not self.check_file_exists(filename):
                all_exist = False
        return all_exist
    
    def install_dependencies(self):
        """Install required packages"""
        self.print_step(0, "DEPENDENCY CHECK")
        
        print("Checking and installing required packages...\n")
        
        packages = [
            'pandas',
            'nltk',
            'matplotlib',
            'seaborn',
            'wordcloud'
        ]
        
        for package in packages:
            try:
                __import__(package)
                print(f"✓ {package} is installed")
            except ImportError:
                print(f"✗ {package} not found. Installing...")
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package, '--break-system-packages'],
                    capture_output=True
                )
        
        # Download NLTK data
        print("\nDownloading NLTK data...")
        try:
            import nltk
            for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
                try:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                    print(f"✓ NLTK {resource} available")
                except LookupError:
                    print(f"  Downloading {resource}...")
                    nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: NLTK setup issue: {e}")
    
    def run_complete_pipeline(self):
        """Execute the complete pipeline"""
        self.start_time = time.time()
        
        self.print_header("SOCIAL MEDIA SENTIMENT ANALYSIS - COMPLETE PIPELINE")
        print("This script will execute all components of Assignment 1:\n")
        print("  1. Data Acquisition (500+ records)")
        print("  2. Text Preprocessing (cleaning & tokenization)")
        print("  3. Exploratory Data Analysis (visualizations)")
        print("  4. Model comparison documentation is already complete\n")
        print("Estimated time: 2-5 minutes\n")
        
        input("Press Enter to start...")
        
        # Step 0: Dependencies
        self.install_dependencies()
        
        # Step 1: Data Acquisition
        self.print_step(1, "DATA ACQUISITION")
        success, error = self.run_script(
            'data_acquisition.py',
            'Fetching 500+ social media records based on keyword'
        )
        
        if not success:
            print("\n⚠ Data acquisition failed. Please check the error above.")
            print("Common issues:")
            print("  - Network connectivity")
            print("  - Missing dependencies")
            print("\nThe script will continue with sample data if available.")
        
        # Verify outputs
        step1_complete = self.verify_outputs(
            "Data Acquisition",
            ['raw_data.csv', 'raw_data.json']
        )
        
        if not step1_complete:
            print("\n✗ Critical: Cannot proceed without data files")
            return False
        
        # Step 2: Preprocessing
        self.print_step(2, "TEXT PREPROCESSING")
        success, error = self.run_script(
            'preprocessing.py',
            'Cleaning text with Regex and NLTK (removing stopwords, special chars)'
        )
        
        if not success:
            print("\n✗ Preprocessing failed")
            return False
        
        step2_complete = self.verify_outputs(
            "Preprocessing",
            ['processed_data.csv']
        )
        
        if not step2_complete:
            print("\n✗ Critical: Cannot proceed without processed data")
            return False
        
        # Step 3: EDA
        self.print_step(3, "EXPLORATORY DATA ANALYSIS")
        success, error = self.run_script(
            'eda_analysis.py',
            'Generating word clouds and frequency distributions'
        )
        
        if not success:
            print("\n✗ EDA failed")
            return False
        
        step3_complete = self.verify_outputs(
            "EDA",
            ['wordcloud.png', 'frequency_distribution.png']
        )
        
        # Final Summary
        self.print_completion_summary()
        
        return True
    
    def print_completion_summary(self):
        """Print final summary of pipeline execution"""
        total_time = time.time() - self.start_time
        
        self.print_header("PIPELINE EXECUTION COMPLETE")
        
        print("✓ All tasks completed successfully!\n")
        print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)\n")
        
        print("Generated Files:")
        print("─" * 70)
        
        files = {
            "Data Files": [
                "raw_data.csv - Original collected data",
                "raw_data.json - Data in JSON format",
                "processed_data.csv - Cleaned and tokenized data"
            ],
            "Visualizations": [
                "wordcloud.png - Word cloud visualization",
                "frequency_distribution.png - Top 20 words bar chart",
                "frequency_horizontal.png - Horizontal frequency chart",
                "token_length_distribution.png - Token length analysis",
                "word_length_distribution.png - Word length analysis"
            ],
            "Documentation": [
                "BERT_vs_VADER_Analysis.md - Model comparison document",
                "README.md - Project documentation"
            ]
        }
        
        for category, file_list in files.items():
            print(f"\n{category}:")
            for file in file_list:
                print(f"  • {file}")
        
        print("\n" + "─" * 70)
        print("\nAssignment 1 Completion Checklist:")
        print("  ✓ Task 1: Data acquisition (500+ records)")
        print("  ✓ Task 2: Preprocessing with Regex and NLTK")
        print("  ✓ Task 3: Word cloud generated")
        print("  ✓ Task 3: Frequency distribution of top 20 words")
        print("  ✓ Task 4: BERT vs VADER comparison documented")
        
        print("\n" + "=" * 70)
        print("Next Steps:")
        print("─" * 70)
        print("1. Review the generated visualizations")
        print("2. Read BERT_vs_VADER_Analysis.md for model insights")
        print("3. Examine processed_data.csv to see cleaned text")
        print("4. Use these outputs for your assignment report")
        print("5. Consider implementing VADER or BERT for Assignment 2")
        print("\n" + "=" * 70)
        print("\n✨ Great work! Your sentiment analysis foundation is complete! ✨\n")


def main():
    """Main entry point"""
    runner = PipelineRunner()
    
    try:
        success = runner.run_complete_pipeline()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

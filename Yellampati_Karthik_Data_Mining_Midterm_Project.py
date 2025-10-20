#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Data Mining Midterm Project - Association Rule Mining

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

import warnings
warnings.filterwarnings('ignore')

# Check if required libraries are installed
try:
    import pandas as pd
    import numpy as np
    from mlxtend.frequent_patterns import apriori, fpgrowth
    from mlxtend.frequent_patterns import association_rules
    print("✓ All required libraries are installed successfully!")
except ImportError as e:
    print(f"✗ Missing library: {e}")
    print("Please run the following installation commands:")
    print("pip install pandas numpy matplotlib seaborn mlxtend")
    exit(1)

print("=" * 60)
print("ASSOCIATION RULE MINING PROJECT - DATA MINING MIDTERM")
print("=" * 60)
print("Initializing Association Rule Mining Project...")

class CompanyDatasetCreator:
    def __init__(self):
        # Define items directly without external files
        self.companies = {
            'Amazon': [
                'echo_dot', 'fire_tv', 'kindle', 'alexa_skills', 'prime_video',
                'amazon_fresh', 'books', 'electronics', 'home_kitchen', 'toys',
                'fashion', 'beauty', 'sports', 'garden', 'office_supplies'
            ],
            'Walmart': [
                'groceries', 'clothing', 'electronics', 'home_decor', 'toys',
                'pharmacy', 'automotive', 'sports_goods', 'furniture', 'jewelry',
                'baby_products', 'pet_supplies', 'cleaning_supplies', 'garden', 'party_supplies'
            ],
            'Nike': [
                'running_shoes', 'basketball_shoes', 'training_shoes', 'sneakers',
                'athletic_shorts', 'sports_bras', 't_shirts', 'hoodies', 'jackets',
                'leggings', 'socks', 'hats', 'backpacks', 'water_bottles', 'accessories'
            ],
            'BestBuy': [
                'laptops', 'tvs', 'headphones', 'smartphones', 'tablets',
                'gaming_consoles', 'cameras', 'smart_home', 'appliances', 'audio_systems',
                'computer_parts', 'cables', 'printers', 'drones', 'wearables'
            ],
            'Target': [
                'home_decor', 'clothing', 'electronics', 'toys', 'groceries',
                'beauty', 'kitchen', 'furniture', 'seasonal', 'school_supplies',
                'party_supplies', 'sports', 'baby', 'pet_supplies', 'health'
            ]
        }
        
        # Verify we have exactly 5 databases as required
        if len(self.companies) != 5:
            raise ValueError("Must have exactly 5 databases as required by project specifications")
        
        print(f"✓ Initialized {len(self.companies)} companies with 15 items each")

    def generate_meaningful_transaction(self, items, company_name, transaction_id):
        """
        Generate transactions with meaningful patterns that will create frequent itemsets
        """
        # Create company-specific patterns that will generate frequent itemsets
        company_patterns = {
            'Amazon': [
                ['electronics', 'home_kitchen', 'toys'],  # High frequency pattern
                ['books', 'kindle', 'electronics'],       # Medium frequency  
                ['amazon_fresh', 'groceries', 'home_kitchen'], # Medium frequency
                ['echo_dot', 'alexa_skills', 'electronics'], # Low frequency
                ['fashion', 'beauty', 'accessories']      # Low frequency
            ],
            'Walmart': [
                ['groceries', 'cleaning_supplies', 'home_decor'], # High frequency
                ['clothing', 'electronics', 'accessories'],       # High frequency
                ['toys', 'baby_products', 'clothing'],           # Medium frequency
                ['pharmacy', 'health', 'groceries'],             # Medium frequency
                ['sports_goods', 'automotive', 'electronics']    # Low frequency
            ],
            'Nike': [
                ['running_shoes', 'athletic_shorts', 't_shirts'], # High frequency
                ['basketball_shoes', 't_shirts', 'socks'],       # High frequency
                ['hoodies', 'leggings', 'sports_bras'],          # Medium frequency
                ['sports_bras', 'training_shoes', 'athletic_shorts'], # Medium frequency
                ['sneakers', 'socks', 'hats']                    # Low frequency
            ],
            'BestBuy': [
                ['electronics', 'laptops', 'computer_parts'],    # High frequency
                ['tvs', 'audio_systems', 'electronics'],         # High frequency
                ['smartphones', 'headphones', 'electronics'],    # Medium frequency
                ['gaming_consoles', 'electronics', 'tvs'],       # Medium frequency
                ['cameras', 'accessories', 'drones']             # Low frequency
            ],
            'Target': [
                ['home_decor', 'kitchen', 'furniture'],          # High frequency
                ['clothing', 'electronics', 'accessories'],      # High frequency
                ['groceries', 'cleaning_supplies', 'home_decor'], # Medium frequency
                ['toys', 'baby', 'clothing'],                    # Medium frequency
                ['school_supplies', 'office', 'electronics']     # Low frequency
            ]
        }
        
        # Determine transaction pattern based on deterministic selection
        pattern_index = (sum(ord(c) for c in company_name) + transaction_id) % len(company_patterns[company_name])
        base_pattern = company_patterns[company_name][pattern_index]
        
        # Add some variation while keeping core patterns
        transaction = list(base_pattern)
        
        # Occasionally add extra items to create more combinations (30% of transactions)
        if (transaction_id % 3) == 0:
            available_extras = [item for item in items if item not in transaction]
            if available_extras:
                extra_index = transaction_id % len(available_extras)
                transaction.append(available_extras[extra_index])
        
        # Ensure all items in transaction exist in our item list
        transaction = [item for item in transaction if item in items]
        
        return sorted(transaction)

    def create_dataset(self, company_name, num_transactions=50): 
        """Create dataset with meaningful patterns that generate frequent itemsets"""
        if company_name not in self.companies:
            raise ValueError(f"Unknown company: {company_name}")
            
        items = self.companies[company_name]
        if not items:
            raise ValueError(f"No items available for {company_name}")
            
        transactions = []
        
        print(f"Creating {num_transactions} meaningful transactions for {company_name}...")
        
        for i in range(num_transactions):
            transaction_items = self.generate_meaningful_transaction(items, company_name, i)
            transactions.append({
                'transaction_id': i + 1,
                'items': ','.join(transaction_items)
            })
        
        return pd.DataFrame(transactions)

class BruteForceMiner:
    def __init__(self, min_support=0.1, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
    
    def load_transactions(self, filename):
        df = pd.read_csv(filename)
        transactions = []
        for _, row in df.iterrows():
            items = row['items'].split(',')
            transactions.append(frozenset(items))
        return transactions
    
    def get_all_items(self, transactions):
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        return sorted(all_items)
    
    def calculate_support(self, itemset, transactions):
        count = 0
        for transaction in transactions:
            if itemset.issubset(transaction):
                count += 1
        return count / len(transactions)
    
    def generate_k_itemsets(self, items, k):
        return [frozenset(combo) for combo in combinations(items, k)]
    
    def find_frequent_itemsets(self, transactions):
        all_items = self.get_all_items(transactions)
        self.frequent_itemsets = {}
        
        k = 1
        while True:
            candidate_itemsets = self.generate_k_itemsets(all_items, k)
            frequent_k_itemsets = []
            
            for itemset in candidate_itemsets:
                support = self.calculate_support(itemset, transactions)
                if support >= self.min_support:
                    frequent_k_itemsets.append((itemset, support))
            
            if not frequent_k_itemsets:
                break
                
            self.frequent_itemsets[k] = frequent_k_itemsets
            k += 1
        
        return self.frequent_itemsets
    
    def generate_association_rules(self, transactions):
        rules = []
        
        for k in range(2, len(self.frequent_itemsets) + 1):
            for itemset, support in self.frequent_itemsets[k]:
                itemset_list = list(itemset)
                
                for i in range(1, len(itemset_list)):
                    for antecedent in combinations(itemset_list, i):
                        antecedent_set = frozenset(antecedent)
                        consequent_set = itemset - antecedent_set
                        
                        antecedent_support = self.calculate_support(antecedent_set, transactions)
                        if antecedent_support > 0:
                            confidence = support / antecedent_support
                            
                            if confidence >= self.min_confidence:
                                rules.append({
                                    'antecedent': antecedent_set,
                                    'consequent': consequent_set,
                                    'support': support,
                                    'confidence': confidence
                                })
        
        return rules

class LibraryMiner:
    def __init__(self, min_support=0.1, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
    
    def load_transactions(self, filename):
        df = pd.read_csv(filename)
        transactions = []
        for _, row in df.iterrows():
            transactions.append(row['items'].split(','))
        return transactions
    
    def prepare_data(self, transactions):
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        return pd.DataFrame(te_ary, columns=te.columns_)
    
    def run_apriori(self, encoded_df):
        frequent_itemsets = apriori(encoded_df, min_support=self.min_support, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        else:
            rules = pd.DataFrame()
        return frequent_itemsets, rules
    
    def run_fpgrowth(self, encoded_df):
        frequent_itemsets = fpgrowth(encoded_df, min_support=self.min_support, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        else:
            rules = pd.DataFrame()
        return frequent_itemsets, rules

class CompleteAnalysisSystem:
    def __init__(self):
        self.datasets = self.find_datasets()
        self.bf_miner = BruteForceMiner()
        
        # Verify we have exactly 5 databases
        if len(self.datasets) != 5:
            print(f"WARNING: Expected 5 databases, found {len(self.datasets)}")
    
    def find_datasets(self):
        datasets = {}
        csv_files = [f for f in os.listdir('.') if f.endswith('_transactions.csv')]
        
        for file in csv_files:
            company = file.replace('_transactions.csv', '').title()
            datasets[company] = file
        
        print(f"Found {len(datasets)} transaction databases")
        return datasets
    
    def display_menu(self):
        print("\n" + "=" * 60)
        print("AVAILABLE DATABASES")
        print("=" * 60)
        for i, (company, filename) in enumerate(self.datasets.items(), 1):
            df = pd.read_csv(filename)
            all_items = set()
            for items_str in df['items']:
                all_items.update(items_str.split(','))
            print(f"{i}. {company} ({len(df)} transactions, {len(all_items)} items)")
        print("=" * 60)
    
    def get_user_parameters(self):
        print("\nSET ANALYSIS PARAMETERS")
        print("-" * 30)
        
        print("RECOMMENDED RANGES:")
        print("Support: 0.05 to 0.3 (lower = more itemsets)")
        print("Confidence: 0.3 to 0.8")
        print("-" * 30)
        
        while True:
            try:
                support = input("Enter minimum support (0.01 to 1.0): ").strip()
                support_val = float(support)
                if 0.01 <= support_val <= 1.0:
                    break
                else:
                    print("Support must be between 0.01 and 1.0")
            except ValueError:
                print("Please enter a valid number (e.g., 0.1)")
        
        while True:
            try:
                confidence = input("Enter minimum confidence (0.01 to 1.0): ").strip()
                confidence_val = float(confidence)
                if 0.01 <= confidence_val <= 1.0:
                    break
                else:
                    print("Confidence must be between 0.01 and 1.0")
            except ValueError:
                print("Please enter a valid number (e.g., 0.5)")
        
        return support_val, confidence_val
    
    def get_dataset_choice(self):
        while True:
            try:
                choice = input(f"\nSelect a database (1-{len(self.datasets)}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(self.datasets):
                    companies = list(self.datasets.keys())
                    selected = companies[choice_num - 1]
                    print(f"Selected: {selected}")
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(self.datasets)}")
            except ValueError:
                print("Please enter a valid number")
    
    def run_brute_force(self, company, support, confidence):
        print("\nRunning Brute Force Algorithm...")
        filename = self.datasets[company]
        
        self.bf_miner.min_support = support
        self.bf_miner.min_confidence = confidence
        
        transactions = self.bf_miner.load_transactions(filename)
        
        start_time = time.perf_counter()
        frequent_itemsets = self.bf_miner.find_frequent_itemsets(transactions)
        rules = self.bf_miner.generate_association_rules(transactions)
        end_time = time.perf_counter()
        
        itemset_count = sum(len(itemsets) for itemsets in frequent_itemsets.values())
        
        # Show detailed results
        print(f"  Transactions analyzed: {len(transactions)}")
        print(f"  Support threshold: {support} (min {int(support * len(transactions))} occurrences)")
        print(f"  Found {itemset_count} frequent itemsets across {len(frequent_itemsets)} sizes")
        print(f"  Generated {len(rules)} association rules")
        
        # Show some example itemsets if found
        if itemset_count > 0:
            print("  Example frequent itemsets:")
            for k, itemsets in list(frequent_itemsets.items())[:2]:  # Show first 2 sizes
                for itemset, supp in itemsets[:3]:  # Show first 3 of each size
                    print(f"    {set(itemset)} (support: {supp:.3f})")
        
        return {
            'time': end_time - start_time,
            'itemsets': itemset_count,
            'rules': len(rules),
            'frequent_itemsets': frequent_itemsets,
            'association_rules': rules
        }
    
    def run_apriori(self, company, support, confidence):
        print("Running Apriori Algorithm...")
        filename = self.datasets[company]
        
        df = pd.read_csv(filename)
        transactions = []
        for _, row in df.iterrows():
            transactions.append(row['items'].split(','))
        
        te = TransactionEncoder()
        encoded_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
        
        start_time = time.perf_counter()
        itemsets = apriori(encoded_df, min_support=support, use_colnames=True)
        if len(itemsets) > 0:
            rules = association_rules(itemsets, metric="confidence", min_threshold=confidence)
        else:
            rules = pd.DataFrame()
        end_time = time.perf_counter()
        
        print(f"  Found {len(itemsets)} frequent itemsets, {len(rules)} rules")
        
        # Show some example itemsets if found
        if len(itemsets) > 0:
            print("  Top 3 frequent itemsets:")
            for _, row in itemsets.head(3).iterrows():
                itemset = list(row['itemsets'])
                print(f"    {itemset} (support: {row['support']:.3f})")
        
        return {
            'time': end_time - start_time,
            'itemsets': len(itemsets),
            'rules': len(rules)
        }
    
    def run_fpgrowth(self, company, support, confidence):
        print("Running FP-Growth Algorithm...")
        filename = self.datasets[company]
        
        df = pd.read_csv(filename)
        transactions = []
        for _, row in df.iterrows():
            transactions.append(row['items'].split(','))
        
        te = TransactionEncoder()
        encoded_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
        
        start_time = time.perf_counter()
        itemsets = fpgrowth(encoded_df, min_support=support, use_colnames=True)
        if len(itemsets) > 0:
            rules = association_rules(itemsets, metric="confidence", min_threshold=confidence)
        else:
            rules = pd.DataFrame()
        end_time = time.perf_counter()
        
        print(f"  Found {len(itemsets)} frequent itemsets, {len(rules)} rules")
        
        # Show some example itemsets if found
        if len(itemsets) > 0:
            print("  Top 3 frequent itemsets:")
            for _, row in itemsets.head(3).iterrows():
                itemset = list(row['itemsets'])
                print(f"    {itemset} (support: {row['support']:.3f})")
        
        return {
            'time': end_time - start_time,
            'itemsets': len(itemsets),
            'rules': len(rules)
        }
    
    def safe_division(self, numerator, denominator):
        min_denominator = 0.000001
        safe_denominator = max(denominator, min_denominator)
        return numerator / safe_denominator
    
    def format_time(self, seconds):
        if seconds < 0.001:
            return f"{seconds * 1000000:.2f} microseconds"
        elif seconds < 1.0:
            return f"{seconds * 1000:.2f} milliseconds"
        else:
            return f"{seconds:.4f} seconds"
    
    def generate_comparison_report(self, results, company, support, confidence):
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ALGORITHM COMPARISON REPORT")
        print("=" * 80)
        print(f"Dataset: {company}")
        print(f"Parameters: Support >= {support}, Confidence >= {confidence}")
        print("=" * 80)
        
        print("PERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Algorithm':<15} {'Time':<20} {'Itemsets':<10} {'Rules':<10} {'Itemsets/sec':<15}")
        print("-" * 80)
        
        for algo_name, result in results.items():
            time_formatted = self.format_time(result['time'])
            itemsets_per_sec = self.safe_division(result['itemsets'], result['time'])
            print(f"{algo_name:<15} {time_formatted:<20} {result['itemsets']:<10} {result['rules']:<10} {itemsets_per_sec:<15.2f}")
        
        print("-" * 80)
        
        # Filter out algorithms with 0 itemsets for meaningful comparisons
        non_zero_results = {k: v for k, v in results.items() if v['itemsets'] > 0}
        
        if non_zero_results:
            fastest_algo = min(non_zero_results.items(), key=lambda x: x[1]['time'])
            most_itemsets = max(non_zero_results.items(), key=lambda x: x[1]['itemsets'])
            most_rules = max(non_zero_results.items(), key=lambda x: x[1]['rules'])
            
            print("KEY FINDINGS:")
            print(f"Fastest Algorithm: {fastest_algo[0]} ({self.format_time(fastest_algo[1]['time'])})")
            print(f"Most Itemsets Found: {most_itemsets[0]} ({most_itemsets[1]['itemsets']} itemsets)")
            print(f"Most Rules Found: {most_rules[0]} ({most_rules[1]['rules']} rules)")
            
            # Performance comparisons
            bf_result = results.get('Brute Force', {})
            apriori_result = results.get('Apriori', {})
            fpgrowth_result = results.get('FP-Growth', {})
            
            if bf_result.get('itemsets', 0) > 0 and apriori_result.get('itemsets', 0) > 0:
                speedup = self.safe_division(bf_result['time'], apriori_result['time'])
                print(f"Apriori is {speedup:.1f}x faster than Brute Force")
            
            if fpgrowth_result.get('itemsets', 0) > 0 and apriori_result.get('itemsets', 0) > 0:
                if fpgrowth_result['time'] < apriori_result['time']:
                    speedup = self.safe_division(apriori_result['time'], fpgrowth_result['time'])
                    print(f"FP-Growth is {speedup:.1f}x faster than Apriori")
                else:
                    speedup = self.safe_division(fpgrowth_result['time'], apriori_result['time'])
                    print(f"Apriori is {speedup:.1f}x faster than FP-Growth")
        else:
            print("KEY FINDINGS:")
            print("No frequent itemsets found with current parameters.")
            print("Try lowering the support threshold (e.g., 0.05-0.2)")
        
        # Consistency check
        itemset_counts = [result['itemsets'] for result in results.values()]
        if len(set(itemset_counts)) == 1:
            if itemset_counts[0] > 0:
                print("✓ All algorithms found the same number of itemsets - RESULTS CONSISTENT")
            else:
                print("⚠ No algorithms found any frequent itemsets")
                print("💡 TIP: Try support=0.05-0.2 and confidence=0.3-0.7")
        else:
            print("⚠ Algorithms found different numbers of itemsets")
            for algo_name, result in results.items():
                print(f"  {algo_name}: {result['itemsets']} itemsets")
        
        # Recommendations
        if non_zero_results:
            fastest_algo_name = min(non_zero_results.items(), key=lambda x: x[1]['time'])[0]
            if fastest_algo_name == 'FP-Growth':
                print("→ RECOMMENDATION: FP-Growth for optimal performance")
            elif fastest_algo_name == 'Apriori':
                print("→ RECOMMENDATION: Apriori provides good balance of speed and readability")
            else:
                print("→ RECOMMENDATION: Brute Force is best for educational purposes")
        else:
            print("→ RECOMMENDATION: Try support = 0.05-0.2 and confidence = 0.3-0.7")
    
    def get_continue_choice(self):
        while True:
            choice = input("\nRun another analysis? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    def start_interactive_analysis(self):
        if not self.datasets:
            print("ERROR: No datasets found! Please check that datasets were created successfully.")
            return
        
        if len(self.datasets) != 5:
            print(f"WARNING: Expected 5 databases, but found {len(self.datasets)}")
        
        print("\n" + "=" * 60)
        print("ASSOCIATION RULE MINING ANALYSIS SYSTEM")
        print("=" * 60)
        print("This system will:")
        print("1. Show available databases")
        print("2. Let you select ONE database")
        print("3. Run Brute Force, Apriori, and FP-Growth algorithms")
        print("4. Provide detailed performance comparisons")
        print("\nRECOMMENDED PARAMETERS FOR MEANINGFUL RESULTS:")
        print("Support: 0.05 to 0.3 (lower = more itemsets)")
        print("Confidence: 0.3 to 0.8")
        print("=" * 60)
        
        analysis_count = 0
        
        while True:
            analysis_count += 1
            print("\n" + "#" * 60)
            print(f"ANALYSIS SESSION #{analysis_count}")
            print("#" * 60)
            
            self.display_menu()
            
            company = self.get_dataset_choice()
            support, confidence = self.get_user_parameters()
            
            print(f"\nStarting analysis with:")
            print(f"  Database: {company}")
            print(f"  Support: {support}")
            print(f"  Confidence: {confidence}")
            print("Running all algorithms...")
            
            results = {}
            
            results['Brute Force'] = self.run_brute_force(company, support, confidence)
            results['Apriori'] = self.run_apriori(company, support, confidence)
            results['FP-Growth'] = self.run_fpgrowth(company, support, confidence)
            
            self.generate_comparison_report(results, company, support, confidence)
            
            if not self.get_continue_choice():
                print("\n" + "=" * 60)
                print("Thank you for using the Association Rule Mining System!")
                print("=" * 60)
                break

# Main execution
if __name__ == "__main__":
    print("Initializing Association Rule Mining Project...")
    
    # Create datasets
    try:
        print("\n" + "=" * 60)
        print("CREATING 5 TRANSACTIONAL DATABASES")
        print("=" * 60)
        
        creator = CompanyDatasetCreator()
        
        created_files = []
        for company in creator.companies.keys():
            df = creator.create_dataset(company, 50)  # Increased to 50 transactions
            filename = f"{company.lower()}_transactions.csv"
            df.to_csv(filename, index=False)
            created_files.append(filename)
            print(f"✓ Created {filename} with {len(df)} transactions")
        
        print(f"\n✓ Successfully created {len(created_files)} CSV databases")
        print("✓ All databases are ready for association rule mining!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
    
    # Run analysis system
    try:
        analysis_system = CompleteAnalysisSystem()
        analysis_system.start_interactive_analysis()
    except Exception as e:
        print(f"Error during analysis: {e}")
    
    print("Project execution completed!")


# In[ ]:





import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_all_metrics(model_name, df, pred_column, K=10):
    mrr_sum = hit_sum = precision_sum = ndcg_sum = 0
    n = len(df)
    
    for _, row in df.iterrows():
        gt = str(row['ground_truth_pmid']).strip()
        # Clean and deduplicate the prediction list
        raw_preds = str(row[pred_column]).split(',')
        pred_ids = []
        for p in raw_preds:
            p_clean = p.strip()
            if p_clean not in pred_ids:
                pred_ids.append(p_clean)
        
        # Limit to Top K
        pred_ids = pred_ids[:K]
        
        if gt in pred_ids:
            # It's a Hit!
            hit_sum += 1
            rank = pred_ids.index(gt) + 1
            
            # MRR
            mrr_sum += 1 / rank
            
            # Precision@K (1 correct answer / K total slots)
            precision_sum += 1 / K
            
            # nDCG (Simplified for single ground truth)
            ndcg_sum += 1 / math.log2(rank + 1)
            
    return {
        "Model": model_name, 
        "MRR@10": mrr_sum / n, 
        "Hit Rate@10": hit_sum / n, 
        "Recall@10": hit_sum / n, # Recall = Hit Rate when there's only 1 GT
        "Precision@10": precision_sum / n,
        "nDCG@10": ndcg_sum / n
    }


def random_select_top_per_category(source_path, file_path, n_per_cat=50):
    df_source = pd.read_csv(source_path)
    df = pd.read_csv(file_path)
    
    combined_df = pd.concat([df_source, df], axis=1)
    print("Available columns:", combined_df.columns.tolist())

    def calculate_lipidbot_score(row):
        gt = str(row['ground_truth_pmid']).strip()
        preds = [p.strip() for p in str(row['lipidbot_predictions']).split(',')]
        try:
            rank = preds.index(gt)
            return 1.0 / (rank + 1)
        except (ValueError, AttributeError):
            return 0.0

    combined_df['lipidbot_score'] = combined_df.apply(calculate_lipidbot_score, axis=1)
    df_shuffled = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # df_sorted = df_shuffled.sort_values(by=['category_id', 'lipidbot_score'], ascending=[True, False])

    final_df = df_shuffled.groupby('category_id').sample(n=100, random_state=42)
    # 3. Save to CSV
    final_df.to_csv("../file/selected_prediction_rows.csv", index=False)

    results = []

    for model_col in ['bm25_predictions', 'pubmedbert_base_predictions', 'lipidbot_predictions']:
        metrics = calculate_all_metrics(model_col, final_df, model_col)
        results.append(metrics)

    # 3. View final comparison
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv("citation_comparison_metrix.csv", index=False)
    print(comparison_df)


final_df = random_select_top_per_category("../file/generated_questions_semantic_full.csv","../file/detailed_predictions.csv", n_per_cat=100)




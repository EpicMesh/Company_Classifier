import pandas as pd
from sentence_transformers import SentenceTransformer,util
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

def read_data(csv_file, use_columns = None):
    if use_columns:
        return pd.read_csv(csv_file, usecols= use_columns)
    return pd.read_csv(csv_file, header= None)

def preprocessData(company_list, insurance_taxonomy):
    total_companies = len(company_list)
    total_labels = len(insurance_taxonomy)

    for i, label in enumerate(insurance_taxonomy):
        if i % 100 == 0:
            print(f"🔄 Processing label {i}/{total_labels}")

        if isinstance(label.get('label'),str):
            label['label_vector'] = model.encode(label['label'], convert_to_tensor=True)

    for i, company in enumerate(company_list):
        if i % 100 == 0:  # Print progress every 50 companies
            print(f"🔄 Processing company {i}/{total_companies}")

        combined_text = ""
        if isinstance(company.get('description'), str):
            combined_text += company['description'] + " "
        if isinstance(company.get('business_tags'), str):
            combined_text += company['business_tags'] + " "
        if isinstance(company.get('category'), str):
            combined_text += company['category'] + " "
        if isinstance(company.get('niche'), str):
            combined_text += company['niche'] + " "

        if combined_text:
            company['combined_text_vector'] = model.encode(combined_text, convert_to_tensor=True)


def classifyCompany(company_list,insurance_taxonomy):

    number_of_unknown_labels = 0
    threshold = 0.2

    label_vectors = torch.stack([label['label_vector'] for label in insurance_taxonomy])

    for company in company_list:

        company_vector = company['combined_text_vector']

        similarity_scores = util.pytorch_cos_sim(company_vector, label_vectors)

        best_score, best_idx = similarity_scores.max(dim=1)
        best_score = best_score.item()
        best_label = insurance_taxonomy[best_idx.item()]['label']

        if best_score > threshold:
            company['label'] = best_label
        else:
            company['label'] = "Unknown"
            number_of_unknown_labels += 1

        company.pop('combined_text_vector', None)

    print(f"❌Unknown labels number:{number_of_unknown_labels}")
    print(f"✅File Saved!")

def main():
    company_list = read_data("company_list.csv",use_columns=["description", "business_tags","sector","category","niche"]).to_dict(orient='records')
    insurance_taxonomy = read_data("insurance_taxonomy.csv", use_columns=[0]).to_dict(orient='records')

    preprocessData(company_list, insurance_taxonomy)
    classifyCompany(company_list, insurance_taxonomy)

    df = pd.DataFrame(company_list)
    df.to_csv('output.csv', index = False)

if __name__ == "__main__":
    main()

import pandas as pd
import spacy
from pandas import read_csv

nlp = spacy.load("en_core_web_lg")


def read_data(csv_file, use_columns = None):
    if use_columns:
        return pd.read_csv(csv_file, usecols= use_columns)
    return pd.read_csv(csv_file, header= None)


def main():
    company_list = read_data("company_list.csv",use_columns=["description", "business_tags","sector","category","niche"]).to_dict(orient='records')
    insurance_taxonomy = read_data("insurance_taxonomy.csv")[0].tolist()
    insurance_taxonomy.remove('label')

    label_vectors = {label: nlp(label.strip().lower()) for label in insurance_taxonomy}

    total_companies = len(company_list)

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
            company['combined_text_vector'] = nlp(combined_text.lower())

    number_of_unknown_labels = 0
    for company in company_list:
        best_label = None
        best_score = 0.0
        for label in insurance_taxonomy:
            label_clean = label.strip().lower()
            if 'business_tags' in company and isinstance(company['business_tags'], str) and label_clean in company['business_tags'].lower():
                best_label = label
                break
            elif 'category' in company and isinstance(company['category'], str) and label_clean in company['category'].lower():
                best_label = label
                break
            elif 'niche' in company and isinstance(company['niche'], str) and label_clean in company['niche'].lower():
                best_label = label
                break
            elif 'combined_text_vector' in company:
                combined_text_doc = company['combined_text_vector']
                label_doc = label_vectors[label]
                similarity_score = combined_text_doc.similarity(label_doc)

                if similarity_score > best_score:
                    best_score = similarity_score
                    best_label = label

        company['label'] = best_label if best_score > 0.7 else "Unknown"
        company.pop('combined_text_vector',None)
        if "Unknown" in company['label']:
            number_of_unknown_labels = number_of_unknown_labels + 1

    print(f"Unknown labels number:{number_of_unknown_labels}")
    print(f"File Saved!")
    df = pd.DataFrame(company_list)
    df.to_csv('output.csv', index = False)

if __name__ == "__main__":
    main()

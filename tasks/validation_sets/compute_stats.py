import os
import torch
import pandas as pd

def main():
    root = '/ssd2/data/body/validation_sets'
    val_names = os.listdir(root)
    
    # Create an empty list to store the data for each validation set
    data = []

    for name in val_names:
        metadata = torch.load(os.path.join(root, name, 'metadata.pt'))
        query_keys = [key for key in metadata.keys() if key.startswith('query')]
        gallery_key = 'gallery'
        query_data = [metadata[key] for key in query_keys]
        gallery_data = metadata[gallery_key]
        
        gallery_subject_ids = set([row[1] for row in gallery_data])
        num_gallery_images = len(gallery_data)
        
        # Initialize a dictionary to store the stats for this validation set
        stats = {
            'name': name,
            'gallery_images': num_gallery_images,
            'gallery_subjects': len(gallery_subject_ids),
        }
        
        all_query_subject_ids = set()
        for i, _query_data in enumerate(query_data):
            query_subject_ids = set([row[1] for row in _query_data])
            all_query_subject_ids.update(query_subject_ids)
            stats[f'query{i+1}_images'] = len(_query_data)
            stats[f'query{i+1}_subjects'] = len(query_subject_ids)
        
        intersection_subject_ids = gallery_subject_ids.intersection(all_query_subject_ids)
        stats['intersection_subjects'] = len(intersection_subject_ids)
        
        data.append(stats)
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data).T
    
    # Display the DataFrame
    print(df)
    
    # Optionally, save the DataFrame to a CSV file
    df.to_csv('validation_set_stats.csv', index=True)

if __name__ == "__main__":
    main()
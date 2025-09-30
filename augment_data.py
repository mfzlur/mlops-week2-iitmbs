import pandas as pd

def augment_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Simple augmentation: duplicate last 10 rows
    augmented_df = pd.concat([df, df.tail(10)], ignore_index=True)
    augmented_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    import sys
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    augment_data(input_path, output_path)

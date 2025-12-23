import pandas as pd

def load_data(input_path: str) -> pd.DataFrame:

    df = pd.read_csv(input_path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    df_prep = df.copy()

    # 1. Drop irrelevant columns
    df_prep = df_prep.drop(columns=['Name', 'Ticket', 'Cabin'])

    # 2. Handle missing values
    df_prep['Age'] = df_prep['Age'].fillna(df_prep['Age'].median())
    df_prep['Embarked'] = df_prep['Embarked'].fillna(
        df_prep['Embarked'].mode()[0]
    )

    # 3. Encode categorical variables
    # Sex: binary encoding
    df_prep['Sex'] = df_prep['Sex'].map({
        'male': 0,
        'female': 1
    })

    # Embarked: one-hot encoding
    df_prep = pd.get_dummies(
        df_prep,
        columns=['Embarked'],
        drop_first=True
    )

    return df_prep

def save_data(df: pd.DataFrame, output_path: str) -> None:

    df.to_csv(output_path, index=False)

def main():
    
    input_path = "namadataset_raw/Titanic-Dataset.csv"
    output_path = "preprocessing/namadataset_preprocessing/titanic_clean.csv"

    df_raw = load_data(input_path)
    df_processed = preprocess_data(df_raw)
    save_data(df_processed, output_path)

    print("Preprocessing completed successfully.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

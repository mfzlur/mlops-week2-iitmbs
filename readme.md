<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

### Step-by-Step Guide to Version IRIS Data and Models with DVC and GCS

I'll guide you through the entire process based on your instructions. We'll use DVC for data and model versioning, GCS as the remote storage for large files, and Git/GitHub for code and metadata. Assume you're working in a GCP environment like Cloud Shell or Vertex AI Workbench, with `gcloud` authenticated and DVC installed (`pip install 'dvc[gs]'`). If not, install DVC first.

We'll create two versions:

- Version 1: Train on raw IRIS data.
- Version 2: Augment the data, retrain, and version.

Finally, demonstrate switching versions. All commands are executable in your terminal.

#### 1. Initialize Git and DVC

Create a new project directory and set up Git/DVC.

- Create and navigate to the project folder:

```
mkdir iris-dvc-project
cd iris-dvc-project
```

- Initialize Git:

```
git init
```

- Initialize DVC:

```
dvc init
git add .dvc
git commit -m "Initialize DVC"
```

- Add GCS as the DVC remote (using your bucket path; adjust if needed):

```
dvc remote add -d myremote gs://mlops-course-bucket-week2-rango-unique/dvc-storage/
git add .dvc/config
git commit -m "Add GCS remote"
```

- (Optional) Authenticate if not already:

```
gcloud auth application-default login
```


This sets up versioning with GCS for data/models.

#### 2. Get Data from GCS and Train Initial Model

Fetch the raw IRIS data from GCS and train a classification model (using RandomForestClassifier for simplicity).

- Download the data locally using `gsutil`:

```
mkdir data
gsutil cp gs://mlops-course-bucket-week2-rango-unique/data/data.csv data/raw_iris.csv
```

- Create a training script (`train_model.py`) to load data, train, and save the model. Copy this code into a file:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df.drop('species', axis=1)  # Assuming 'species' is the target column
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")
    
    joblib.dump(clf, model_path)

if __name__ == '__main__':
    import sys
    data_path = sys.argv[^1]
    model_path = sys.argv[^2]
    train_model(data_path, model_path)
```

- Execute the training script:

```
mkdir models
python train_model.py data/raw_iris.csv models/model_v1.pkl
```

This trains on the raw data and saves `model_v1.pkl`. It should print the accuracy.


#### 3. Track Model, Data, and Push to GitHub and GCS

Version the raw data and initial model with DVC, then push.

- Add data and model to DVC:

```
dvc add data/raw_iris.csv models/model_v1.pkl
git add data/raw_iris.csv.dvc models/model_v1.pkl.dvc .gitignore
git commit -m "Track raw data and initial model with DVC"
```

- Push data/model to GCS:

```
dvc push
```

- Set up GitHub remote (create a repo on GitHub first, then):

```
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

This pushes code and DVC metadata to GitHub, while large files go to GCS.


#### 4. Augment the Dataset

Create a script to augment the data (e.g., by duplicating rows to simulate additions).

- Create an augmentation script (`augment_data.py`). Copy this code into a file:

```python
import pandas as pd

def augment_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Simple augmentation: duplicate last 10 rows
    augmented_df = pd.concat([df, df.tail(10)], ignore_index=True)
    augmented_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    import sys
    input_path = sys.argv[^1]
    output_path = sys.argv[^2]
    augment_data(input_path, output_path)
```

- Run the script to create augmented data:

```
python augment_data.py data/raw_iris.csv data/augmented_iris.csv
```


#### 5. Retrain on Augmented Data, Version, and Push

Train a new model on the augmented data and version it.

- Execute training on augmented data:

```
python train_model.py data/augmented_iris.csv models/model_v2.pkl
```

This creates `model_v2.pkl` with potentially different accuracy.
- Add the augmented data and new model to DVC (this creates a new version):

```
dvc add data/augmented_iris.csv models/model_v2.pkl
git add data/augmented_iris.csv.dvc models/model_v2.pkl.dvc
git commit -m "Track augmented data and retrained model with DVC"
```

- Push to GCS and GitHub:

```
dvc push
git push origin main
```


Now you have two versions tracked: raw (v1) and augmented (v2).

#### 6. Switch Between Data and Model Versions

Use Git to switch commits and DVC to sync data/models.

- View commit history:

```
git log --oneline
```

Note hashes: e.g., `<v1-hash>` for initial commit, `<v2-hash>` for augmented.
- Switch to Version 1 (raw data and initial model):

```
git checkout <v1-hash>
dvc checkout
```

This reverts your workspace to raw_iris.csv and model_v1.pkl from GCS.
- Verify:

```
ls data/ models/
dvc status  # Should show up-to-date
python train_model.py data/raw_iris.csv models/model_v1.pkl  # Optional: Retrain to confirm
```

- Switch back to Version 2 (augmented data and retrained model):

```
git checkout main  # Or <v2-hash>
dvc checkout
```

Now you have augmented_iris.csv and model_v2.pkl.

This demonstrates traversing versions effortlessly. If you pull in a new environment (`git clone` + `dvc pull`), it fetches from GitHub/GCS automatically. Everything is tracked—repeat for more versions as needed.

<div align="center">⁂</div>

[^1]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/d78b29147fd46ab3e3b84395bd515c55/ff0c6101-afb8-4b02-804e-1ae2d3a7435e/2be78259.py

[^2]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/d78b29147fd46ab3e3b84395bd515c55/ff0c6101-afb8-4b02-804e-1ae2d3a7435e/b2714d52.py


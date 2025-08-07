import joblib
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# 1. Load the trained model
model = joblib.load("random_forest_model.pkl")

# 2. Load test data
test_data = pd.read_csv("test_features_normalized.csv")
X_test = test_data.drop(columns=["file_path", "label"])
y_test = test_data["label"]

# 3. Initialize LIME explainer
explainer = LimeTabularExplainer(
    training_data=np.array(X_test),
    feature_names=X_test.columns.tolist(),
    class_names=["HC", "PD"],  # change if you have different labels
    mode="classification"
)

# 4. Choose one instance to explain (e.g., the first sample)
instance_index = 0
instance = X_test.iloc[instance_index].to_numpy()

# 5. Generate explanation for this instance
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba
)

# 6. Show explanation in the console
print(f"Explanation for instance {instance_index}:")
print(exp.as_list())

# 7. Save the LIME explanation plot
fig = exp.as_pyplot_figure()
plt.title(f"LIME Explanation for Sample {instance_index}")
plt.savefig("source/reports/lime_explanation_sample.png")
plt.close()
print("LIME Explanation saved to 'source/reports/lime_explanation_sample.png")


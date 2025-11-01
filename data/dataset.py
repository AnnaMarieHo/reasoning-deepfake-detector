from datasets import load_dataset, concatenate_datasets
import os
from PIL import Image
import pandas as pd

# Load datasets
dd_qva = load_dataset("iamjinchen/DD-VQA", split="train")

dd_qva = dd_qva.select(range(min(5000, len(dd_qva))))
dd_qva = dd_qva.map(lambda x: {"text": x["text"]}, remove_columns=[c for c in dd_qva.column_names if c not in ["image", "text"]])

sid = load_dataset("saberzl/SID_Set_description")
sid_train = sid['train'].rename_column("description", "text")
sid_val = sid['validation'].rename_column("description", "text")

sid_train = sid_train.remove_columns([c for c in sid_train.column_names if c not in ["image", "label", "text"]])
sid_val = sid_val.remove_columns([c for c in sid_val.column_names if c not in ["image", "label", "text"]])

sid_label_feature = sid['train'].features['label']

# Wrap in a Features object
dd_qva = dd_qva.map(
    lambda x: {"label": sid_label_feature.str2int("tampered")},
    features=Features({
        "image": dd_qva.features["image"],
        "text": dd_qva.features["text"],
        "label": sid_label_feature
    })
)

combined_ds = concatenate_datasets([dd_qva, sid_train, sid_val])
print(combined_ds)


image_dir = './data/combined/images'
os.makedirs(image_dir, exist_ok=True)

image_filenames = []
texts = []
labels = []

for i, item in enumerate(combined_ds):
    # Save image
    img = item['image'].convert("RGB")
    filename = f"{i:05d}.png"
    img.save(os.path.join(image_dir, filename))

    # Append text
    text = item.get("text", "")
    texts.append(text)
    image_filenames.append(filename)
    label = item.get("label", "")
    labels.append(label)

df_model = pd.DataFrame({
    "image": image_filenames,
    "text": texts,
    "label": labels
})

df_model["text"] = df_model["text"].apply(lambda s: s.encode('utf-8', errors='replace').decode('utf-8'))

# Make sure label column is numeric
df_model["label"] = pd.to_numeric(df_model["label"], errors='coerce').astype("Int64")

os.makedirs('./data/combined', exist_ok=True)
df_model.to_json('./data/combined/final_combined_model_input.json', orient='records', lines=True, force_ascii=False)

print(df_model.head())

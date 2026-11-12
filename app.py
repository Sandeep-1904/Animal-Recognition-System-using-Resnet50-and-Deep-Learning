
import spaces
import torch
import gradio as gr

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from datetime import datetime
from transformers import (
    AutoImageProcessor,
    ResNetForImageClassification
)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==========================================
# 1. LOAD PRETRAINED MODEL
# ==========================================

MODEL_NAME = "microsoft/resnet-50"

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = ResNetForImageClassification.from_pretrained(
    MODEL_NAME
).to("cuda")

model.eval()


# ==========================================
# 2. ANIMAL CATEGORIES
# ==========================================

ANIMAL_KEYWORDS = [
    "dog", "cat", "wolf", "fox", "tiger", "lion",
    "leopard", "cheetah", "elephant", "bear", "panda",
    "horse", "zebra", "giraffe", "deer", "cow",
    "sheep", "goat", "pig", "rabbit", "hamster",
    "squirrel", "monkey", "ape", "gorilla", "chimpanzee",
    "kangaroo", "koala", "camel", "donkey", "ox",
    "buffalo", "bison", "antelope", "gazelle", "rhino",
    "hippopotamus", "whale", "dolphin", "shark", "fish",
    "eagle", "owl", "parrot", "penguin", "flamingo",
    "duck", "goose", "chicken", "hen", "rooster",
    "turkey", "ostrich", "peacock", "swan", "sparrow",
    "crow", "pigeon", "woodpecker", "snake", "python",
    "cobra", "lizard", "turtle", "tortoise", "crocodile",
    "alligator", "frog", "toad", "salamander",
    "butterfly", "bee", "ant", "spider", "scorpion",
    "crab", "lobster", "octopus", "snail", "worm",
    "starfish", "jellyfish", "sea lion", "seal", "otter",
    "bat", "hedgehog", "porcupine", "raccoon", "skunk",
    "meerkat", "mongoose", "weasel", "badger", "sloth",
    "armadillo", "platypus", "lemur", "orangutan"
]


# ==========================================
# 3. ANIMAL LABEL FILTER
# ==========================================

def is_animal(label):

    label = label.lower().replace("_", " ")

    return any(
        keyword in label
        for keyword in ANIMAL_KEYWORDS
    )


# ==========================================
# 4. ANIMAL RECOGNITION
# ==========================================

@spaces.GPU(duration=30)
def recognize_animal(image, history):

    if image is None:
        raise gr.Error("Please upload an image.")

    image = image.convert("RGB")

    # Image preprocessing
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    # Model prediction
    with torch.inference_mode():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    # Sort predictions
    sorted_indices = torch.argsort(
        probabilities,
        descending=True
    )

    animal_results = {}

    # Filter animal predictions
    for index in sorted_indices:

        label = model.config.id2label[
            index.item()
        ]

        score = probabilities[index].item()

        if is_animal(label):

            animal_results[label] = score

        if len(animal_results) == 5:
            break

    # No animal detected
    if not animal_results:

        return (
            {},
            """
            ## ❌ No Recognized Animal Detected

            Please upload a clear animal image.
            """,
            history or []
        )

    # Best prediction
    best_label = next(iter(animal_results))

    confidence = animal_results[best_label] * 100

    # Confidence threshold
    if confidence < 20:

        return (
            animal_results,
            """
            ## ⚠️ Low Confidence

            The image does not contain a confidently
            recognized animal category.

            Try uploading a clearer animal image.
            """,
            history or []
        )

    # Display result
    result = f"""
    ## 🐾 Animal Recognition Result

    **Animal:** {best_label.title()}

    **Confidence:** {confidence:.2f}%

    **Status:** Animal category recognized
    """

    # ======================================
    # SAVE PREDICTION HISTORY
    # ======================================

    history = history or []

    timestamp = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    new_record = [
        len(history) + 1,
        best_label.title(),
        f"{confidence:.2f}%",
        timestamp
    ]

    # Add newest prediction at the top
    history = [new_record] + history

    return animal_results, result, history


# ==========================================
# 5. CLEAR PREDICTION HISTORY
# ==========================================

def clear_history():

    return [], []


# ==========================================
# 6. CONFUSION MATRIX EVALUATION
# ==========================================

def evaluate_model(true_labels, predicted_labels):

    if not true_labels or not predicted_labels:

        raise gr.Error(
            "Please enter actual and predicted labels."
        )

    # Accept comma-separated or line-separated labels
    actual = [
        x.strip().lower()
        for x in re.split(r"[,\n]+", true_labels)
        if x.strip()
    ]

    predicted = [
        x.strip().lower()
        for x in re.split(r"[,\n]+", predicted_labels)
        if x.strip()
    ]

    # Validate input
    if len(actual) != len(predicted):

        raise gr.Error(
            "Actual and predicted labels must have equal numbers of entries."
        )

    if len(actual) < 2:

        raise gr.Error(
            "Please enter at least two evaluation samples."
        )

    # Get all unique animal classes
    labels = sorted(
        set(actual) | set(predicted)
    )

    # Calculate confusion matrix
    cm = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    # Calculate performance metrics
    accuracy = accuracy_score(
        actual,
        predicted
    )

    precision = precision_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    # ======================================
    # CONFUSION MATRIX VISUALIZATION
    # ======================================

    fig, ax = plt.subplots(
        figsize=(max(8, len(labels) * 1.2),
                 max(6, len(labels) * 0.9))
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Animal Recognition - Confusion Matrix",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=12
    )

    ax.set_ylabel(
        "Actual Label",
        fontsize=12
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    # ======================================
    # CLASSIFICATION REPORT
    # ======================================

    report = classification_report(
        actual,
        predicted,
        labels=labels,
        zero_division=0
    )

    # ======================================
    # PERFORMANCE SUMMARY
    # ======================================

    metrics = f"""
    ## 📊 Model Performance Results

    | Evaluation Metric | Score |
    |---|---:|
    | Accuracy | {accuracy * 100:.2f}% |
    | Precision (Macro) | {precision * 100:.2f}% |
    | Recall (Macro) | {recall * 100:.2f}% |
    | F1-Score (Macro) | {f1 * 100:.2f}% |

    **Total Samples:** {len(actual)}

    **Animal Classes:** {len(labels)}

    **Correct Predictions:** {sum(
        a == p for a, p in zip(actual, predicted)
    )}

    ### Classification Report

    ```text
    {report}
    ```
    """

    return fig, metrics


# ==========================================
# 7. GRADIO APPLICATION
# ==========================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Animal Recognition System"
) as demo:

    # Main heading
    gr.Markdown(
        """
        # 🐾 Animal Recognition System

        An AI-powered image classification application
        using Deep Learning and ResNet-50.

        Upload an image to recognize an animal and
        view its prediction confidence.
        """
    )

    # Session history
    history_state = gr.State([])

    # ======================================
    # ANIMAL RECOGNITION SECTION
    # ======================================

    gr.Markdown("## 🔍 Animal Recognition")

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Animal Image"
            )

            predict_button = gr.Button(
                "🔍 Recognize Animal",
                variant="primary"
            )

            clear_button = gr.ClearButton(
                components=[image_input],
                value="🗑️ Clear Image"
            )

        with gr.Column():

            prediction_output = gr.Label(
                num_top_classes=5,
                label="Animal Predictions"
            )

            result_output = gr.Markdown()

    # ======================================
    # PREVIOUS PREDICTION HISTORY
    # ======================================

    gr.Markdown("---")

    gr.Markdown(
        """
        ## 📋 Previous Prediction History

        View your previous animal recognition results.
        """
    )

    history_table = gr.Dataframe(
        headers=[
            "S.No",
            "Animal Name",
            "Confidence",
            "Date & Time"
        ],

        datatype=[
            "number",
            "str",
            "str",
            "str"
        ],

        value=[],
        interactive=False,
        label="Recognition History",
        wrap=True
    )

    clear_history_button = gr.Button(
        "🗑️ Clear Prediction History",
        variant="stop"
    )

    # ======================================
    # CONFUSION MATRIX SECTION
    # ======================================

    gr.Markdown("---")

    gr.Markdown(
        """
        ## 📊 Confusion Matrix & Model Evaluation

        Enter actual animal labels and corresponding
        predicted animal labels from a labeled test set.

        Separate each label using a comma or a new line.
        Both lists must contain the same number of samples.
        """
    )

    with gr.Row():

        with gr.Column():

            true_labels_input = gr.Textbox(
                label="Actual Animal Labels",
                placeholder="cat,dog,lion,elephant,cat",
                lines=5
            )

        with gr.Column():

            predicted_labels_input = gr.Textbox(
                label="Predicted Animal Labels",
                placeholder="cat,dog,tiger,elephant,cat",
                lines=5
            )

    confusion_button = gr.Button(
        "📊 Generate Confusion Matrix",
        variant="primary"
    )

    confusion_output = gr.Plot(
        label="Confusion Matrix"
    )

    metrics_output = gr.Markdown()

    # ======================================
    # PREDICTION EVENT
    # ======================================

    predict_button.click(
        fn=recognize_animal,

        inputs=[
            image_input,
            history_state
        ],

        outputs=[
            prediction_output,
            result_output,
            history_state
        ]
    ).then(
        fn=lambda history: history,

        inputs=history_state,

        outputs=history_table
    )

    # ======================================
    # CLEAR HISTORY EVENT
    # ======================================

    clear_history_button.click(
        fn=clear_history,

        inputs=[],

        outputs=[
            history_state,
            history_table
        ]
    )

    # ======================================
    # CONFUSION MATRIX EVENT
    # ======================================

    confusion_button.click(
        fn=evaluate_model,

        inputs=[
            true_labels_input,
            predicted_labels_input
        ],

        outputs=[
            confusion_output,
            metrics_output
        ]
    )

    # ======================================
    # FOOTER
    # ======================================

    gr.Markdown(
        """
        ---

        ### Project Information

        - **Project:** Animal Recognition System
        - **Model:** ResNet-50
        - **Framework:** PyTorch
        - **Interface:** Gradio
        - **Deployment:** Hugging Face Spaces
        - **Hardware:** Hugging Face ZeroGPU
        - **License:** MIT
        """
    )


# ==========================================
# 8. LAUNCH APPLICATION
# ==========================================

if __name__ == "__main__":

    demo.launch()

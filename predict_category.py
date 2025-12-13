import joblib
import pandas as pd
import os

# IMPORTANT: This import makes sure the module exists for unpickling
import feature_utils  # noqa: F401


def main():
    model_path = os.path.join("model", "product_category_model.pkl")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("✅ Model loaded successfully!")
    print("Type 'exit' to quit.\n")

    while True:
        title = input("Enter product title: ").strip()
        if title.lower() == "exit":
            print("Exiting...")
            break

        user_input = pd.DataFrame([{"product_title": title}])
        pred = model.predict(user_input)[0]
        print(f"✅ Predicted category: {pred}\n" + "-" * 40)


if __name__ == "__main__":
    main()

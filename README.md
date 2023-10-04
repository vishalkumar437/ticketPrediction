# Railway Ticket Prediction and Confirmation Percentage

This project aims to predict the confirmation percentage of railway tickets using various machine learning algorithms such as Random Forest, Decision Tree, XGBoost, and Naive Bayes. Additionally, it provides a user-friendly interface using Flask for easy interaction.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Algorithms Used](#algorithms-used)
- [Dataset](#dataset)
- [Model Comparison](#model-comparison)
- [UI Using Flask](#ui-using-flask)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

1. Clone the repository:

```
git clone https://github.com/your_username/railway-ticket-prediction.git
cd railway-ticket-prediction
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Ensure you have the necessary dataset (See [Dataset](#dataset) section).
2. Train and evaluate the models using the provided scripts.
3. Start the Flask app to use the UI for ticket prediction.

## Algorithms Used

This project uses the following machine learning algorithms:

- Random Forest
- Decision Tree
- XGBoost
- Naive Bayes

The performance of each algorithm is compared based on their accuracy in predicting ticket confirmation.

## Dataset

The dataset used for this project contains historical railway ticket booking and confirmation data. It includes features like journey date, source, destination, class, etc. Ensure you have the dataset in the `data` directory.

## Model Comparison

To compare the models, execute the `model_comparison.py` script. This will train and evaluate each algorithm, displaying their respective accuracies.

```
python model_comparison.py
```

## UI Using Flask

To run the Flask app for user interaction, execute the following command:

```
python app.py
```

This will start a local server. Visit `http://localhost:5000` in your web browser to use the UI for ticket prediction.

## Contributing

Feel free to contribute by opening issues or creating pull requests. Follow the [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the [MIT License](LICENSE).

---

Happy ticket prediction! 🚆

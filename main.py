import numpy as np

import prediction_model


# for test purpose(example)
def main():
    model = prediction_model.PredictionModel()
    sample = np.array([[1] * 29 for _ in range(6)])
    print(model.predict_next_month_sum_distribution(sample).shape)


if __name__ == '__main__':
    main()

from part2_claim_classifier import *
from part3_pricing_model import *

def optimising_hyperparam_for_part2(raw_data):
    train, test = train_test_split(raw_data, test_size=0.2)
    optimisation_results=ClaimClassifierHyperParameterSearch(train, prefix="p2_")
    saveCurrentNetworkResults(optimisation_results,"p2_optimised_results")
    print(optimisation_results)
    return optimisation_results

def optimising_hyperparam_for_part3(raw_data):
    train, test = train_test_split(raw_data, test_size=0.2)
    pricing_model = PricingModel(calibrate_probabilities=False)
    y_train = train["made_claim"]
    train_claim_severity = train["claim_amount"]
    pricing_model.fit(train, y_train, train_claim_severity)
    pricing_model.save_model()

    X_train = train.drop(columns=["claim_amount", "made_claim"])

    predictions = pricing_model.predict_claim_probability(X_train)
    premiums = pricing_model.predict_premium(X_train)
    scale = pricing_model.set_premium_scale(premiums, train_claim_severity)
    print(scale)
    print(np.sum(premiums) / np.sum(test["claim_amount"]))

def training_and_testing_classifier(raw_data):
    train, test = train_test_split(raw_data, test_size=0.2)
    classifier = ClaimClassifier()
    classifier.fit(train)
    classifier.save_model()

    X_test = test.drop(columns=["claim_amount", "made_claim"])
    y_test = test["made_claim"]
    y_test = y_test.to_numpy()
    y_test = y_test.astype(float)
    predictions = classifier.predict(X_test)
    classifier.evaluate_architecture(predictions, y_test)
    classifier.save_model()
    return classifier

def training_and_testing_initial_classifier(raw_data, sampling_method):
    train, test = train_test_split(raw_data, test_size=0.2)
    classifier = ClaimClassifier()
    classifier.fit(train, layers=[nn.Linear(9, 12), nn.Tanh(), nn.Linear(12, 1), nn.Sigmoid()],
                   sampling=sampling_method)

    X_test = test.drop(columns=["claim_amount", "made_claim"])
    y_test = test["made_claim"]
    y_test = y_test.to_numpy()
    y_test = y_test.astype(float)
    predictions = classifier.predict(X_test)
    print("Evaluation using testing dataset: ")
    classifier.evaluate_architecture(predictions, y_test)

def training_and_testing_for_pricing_model(raw_data):

    # Splitting into training(+validation) datasets and test datasets
    train, test = train_test_split(raw_data, test_size=0.2)

    # Create classifier object
    pricing_model = PricingModel(calibrate_probabilities=False)

    # Use classifier to fit
    y_train = train["made_claim"]
    train_claim_severity = train["claim_amount"]
    pricing_model.fit(train, y_train, train_claim_severity)
    pricing_model.save_model()

    # #Converting test data into useable formats
    X_test = test.drop(columns=["claim_amount", "made_claim"])
    y_test = test["made_claim"]
    y_test = y_test.to_numpy()
    y_test = y_test.astype(float)

    print("Predicting...")
    predictions = pricing_model.predict_claim_probability(X_test)

    pricing_model.base_classifier.evaluate_architecture(predictions, y_test)
    premiums = pricing_model.predict_premium(X_test)

    return (pricing_model)

def data_review(raw_data):
    count_class_0, count_class_1 = raw_data.made_claim.value_counts()
    ratio = count_class_0/count_class_1
    print("Number of samples with no claims: ", count_class_0)
    print("Number of samples with claims: ", count_class_1)
    print("ratio of no claims to claims: %0.2f:1" % ratio)

if __name__ == "__main__":

    #Import data via Dataset class
    print("Loading the datasets...")
    dat_p2 = pd.read_csv('part2_training_data.csv')
    dat_p3 = pd.read_csv('part3_training_data.csv')
    training_and_testing_for_pricing_model(dat_p3)





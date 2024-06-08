b = XGBClassifier(random_state=42)

one_hot = OneHotEncoder()
encoded_train_categoricals = one_hot.fit_transform(train[categorical_columns])
encoded_train_categoricals = pd.DataFrame(data=encoded_train_categoricals.toarray(), columns=one_hot.get_feature_names_out())

encoded_train = pd.concat((train.drop(columns=categorical_columns), encoded_train_categoricals), axis=1)
encoded_train_X = encoded_train.drop(columns='Target')
encoded_train_y = encoded_train.Target

b.fit(encoded_train_X, encoded_train_y)
xgboost.plot_importance(b, max_num_features=37)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import simplejson as json
import urllib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

log = {}

def model(bdr_pivot, learning_rates=[0.1], n_estimators=[200], max_depth=[2],
          test_size=0.33):
    # bdr_pivot = pd.DataFrame(braindr_pivot)
    X = bdr_pivot[[c for c in bdr_pivot.columns if c not in ['plain_average', 'truth']]].values
    y = bdr_pivot.truth.values
    log["X_shape"] = X.shape
    log['y_shape'] = y.shape

    seed = 7
    # test_size = 0.33

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=seed,
                                                        stratify=y)
    log['X_train_shape'] = X_train.shape
    # make sure everyone has a vote in the train and test
    assert(np.isfinite(X_train).sum(0).all())
    assert(np.isfinite(X_test).sum(0).all())

    model = XGBClassifier()

    # parameters to tune
    # learning_rate = [0.1]
    # n_estimators = [200]
    # max_depth = [2]  # , 6, 8]

    # run the grid search
    param_grid = dict(learning_rate=learning_rates,
                      max_depth=max_depth,
                      n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss",
                               n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X_train, y_train)

    # results
    log["Best: %f using %s"] = (grid_result.best_score_,
                                grid_result.best_params_)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #    log["%f (%f) with: %r"] = (mean, stdev, param)

    # make predictions for test data
    # y_pred = model.predict(X_test)
    y_pred = grid_result.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    # log["Accuracy: %.2f%%"] = (accuracy * 100.0)

    y_pred_prob = grid_result.predict_proba(X_test)[:, 1]
    log['y_pred_prob'] = y_pred_prob.tolist()
    log["y_test"] = y_test.tolist()

    B = grid_result.best_estimator_.get_booster()
    fscores = B.get_fscore()
    fdf = pd.DataFrame([fscores]).T.rename(columns={0: 'F'})
    not_col = ['plain_average', 'truth']
    users = [c for c in bdr_pivot.columns if c not in not_col]
    fdf['user'] = fdf.index.map(lambda x: users[int(x[1:])])
    fdf.sort_values('F', inplace=True)
    log['user_importance'] = fdf[::-1].to_json(orient='records')
    return grid_result


def main(braindr_data, pass_labels, fail_labels, learning_rates=[0.1],
         n_estimators=[200], max_depth=[2], test_size=0.33):
    if braindr_data.startswith('http'):
        braindr_df = pd.read_csv(braindr_data)
    else:
        braindr_df = pd.read_table(braindr_data)

    braindr_df['subject_name'] = braindr_df.image_id.map(lambda x: x.split('__')[0])
    braindr_df_pass_subset = braindr_df[braindr_df.subject_name.isin(pass_labels)]
    braindr_df_fail_subset = braindr_df[braindr_df.subject_name.isin(fail_labels)]
    braindr_df_pass_subset['truth'] = 1
    braindr_df_fail_subset['truth'] = 0

    braindr_subset = braindr_df_pass_subset.append(braindr_df_fail_subset,
                                                   ignore_index=True)

    # count users contributions
    user_counts = braindr_subset.groupby('username').apply(lambda x: x.shape[0])
    username_keep = user_counts[user_counts >= user_counts.describe()['75%']].index.values
    bdr = braindr_subset[braindr_subset.username.isin(username_keep)]

    bdr_pivot = braindr_subset.pivot_table(columns="username", index='image_id',
                                           values='vote', aggfunc=np.mean)

    uname_img_counts = pd.DataFrame()
    for uname in bdr_pivot.columns:
        uname_img_counts.loc[uname, 'counts'] = (pd.isnull(bdr_pivot[uname]) == False).sum()

    username_keep = uname_img_counts[uname_img_counts.counts >= uname_img_counts.describe().loc['75%']['counts'] ]
    username_keep = username_keep.index.values

    bdr = braindr_subset[braindr_subset.username.isin(username_keep)]
    bdr_pivot = bdr.pivot_table(columns="username", index='image_id',
                                values='vote', aggfunc=np.mean)
    truth_vals = bdr.groupby('image_id').apply(lambda x: x.truth.values[0])
    bdr_pivot['truth'] = truth_vals

    plain_avg = bdr_pivot[bdr_pivot.columns[:-1]].mean(1)
    bdr_pivot['plain_average'] = plain_avg
    log['bdr_pivot'] = bdr_pivot.to_json(orient='columns')

    grid_result = model(bdr_pivot, learning_rates=learning_rates,
                        n_estimators=n_estimators, max_depth=max_depth,
                        test_size=test_size)

    modelUsers = [c for c in bdr_pivot.columns if c not in ['plain_average',
                                                            'truth']]
    braindr_full_pivot = braindr_df[braindr_df.username.isin(modelUsers)]\
    .pivot_table(columns='username', index='image_id',
                 values='vote', aggfunc=np.mean)
    # braindr_full_pivot = braindr_full_pivot[modelUsers]
    log['braindr_full_pivot_shape'] = braindr_full_pivot.shape

    X_all = braindr_full_pivot.values
    y_all_pred = grid_result.best_estimator_.predict_proba(X_all)
    # model.predict_proba(X_all)

    plain_avg = braindr_full_pivot.mean(1)
    braindr_full_pivot['average_label'] = plain_avg
    braindr_full_pivot['xgboost_label'] = y_all_pred[:, 1]

    log['output'] = braindr_full_pivot.to_json(orient='columns')
    return log  # braindr_full_pivot.to_json(orient='columns')


def handle(st):
    inp = json.loads(st)
    res = main(**inp)
    print(json.dumps(res))

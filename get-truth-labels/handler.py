import pandas as pd
import numpy as np
import simplejson as json
import urllib

log = {}


def download_image(inp, out):
    urllib.urlretrieve(inp, out)
    return out


def tidy_df(mind_df):
    mind_df_tidy = pd.DataFrame()

    for i, row in mind_df.iterrows():
        for entry in row.quality_vote:
            entry['name'] = row['name']
            try:
                entry['vote'] = int(entry['quality_check']['QC'])
                entry['notes'] = entry['quality_check']['notes_QC']
                # some people used emails as username
                entry['checkedBy'] = entry['checkedBy'].split('@')[0]
                mind_df_tidy = mind_df_tidy.append(entry, ignore_index=True)
            except KeyError:
                pass

    def calc_score(index):
        vote = mind_df_tidy.loc[index].vote
        conf = mind_df_tidy.loc[index].confidence
        if vote:
            return abs(conf)
        else:
            return -1*abs(conf)

    mind_df_tidy['score'] = mind_df_tidy.index.map(calc_score)
    return mind_df_tidy





def main(input_data, truth_users):
    if input_data.startswith('http'):
        mc_file = download_image(input_data, 'mindcontrol_data.json')
        with open(mc_file, 'r') as f:
            data = [eval(l) for l in f.readlines()]
    else:
        data = [eval(l) for l in input_data.split('\n')]

    mind_df = pd.DataFrame(data)
    mind_df = mind_df[mind_df.entry_type == 'T1w']
    mind_df_tidy = tidy_df(mind_df)

    passes = mind_df_tidy[mind_df_tidy.score >= 4]
    fails = mind_df_tidy[mind_df_tidy.score <= -4]

    log['passes_shape'] = passes.shape
    log['fails_shape'] = fails.shape

    # remove any images with defacing errors
    not_including = []

    for i, row in fails[fails.notes != ""].iterrows():
        if 'defac' in row.notes.lower() or 'slice' in row.notes.lower():
            not_including.append(row['name'])

    # make it unique
    not_including = list(set(not_including))

    log['not_including'] = not_including

    # filter them out from the fails
    fails = fails[~fails['name'].isin(not_including)]

    ak_pass = passes[passes.checkedBy.isin(truth_users)]['name'].values
    ak_fail = fails[fails.checkedBy.isin(truth_users)]['name'].values
    ak_N = min(ak_pass.shape[0], ak_fail.shape[0])

    log['truth_passes_shape'] = ak_pass.shape
    log['truth_fail_shape'] = ak_fail.shape

    np.random.seed(0)

    idx = np.arange(ak_pass.shape[0])
    np.random.shuffle(idx)
    ak_pass_subset = ak_pass.iloc[idx[:ak_N]]

    passing_names = ak_pass_subset # ['name'].values
    failing_names = ak_fail # ['name'].values

    log['passing_names'] = passing_names.tolist()
    log['failing_names'] = failing_names.tolist()
    log['tidy_df'] = mind_df_tidy
    return log


def handle(st):
    inp = json.loads(st)
    log = main(**inp)
    print(json.dumps(log))

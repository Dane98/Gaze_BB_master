import pandas as pd
from readbbtxt import readbbtxt as readvis
from readbbtxt_inv import readbbtxt as readinv
import cv2

datafolder = './data_fin/'
datafilevis = 'pixel_position_vis.txt'
datafileinvis = 'pixel_position_invis_new.txt'

datavis = readvis(datafolder+datafilevis)
# remove .png extension from filenames
datavis['file'] = datavis['file'].apply(lambda x: x[:-4])

datainvis = readinv(datafolder+datafileinvis)
# remove .png extension from filenames
datainvis['file'] = datainvis['file'].apply(lambda x: x[:-4])

annotations = pd.read_csv('annotations.csv')
annotations_double = pd.read_csv('annotations_double.csv')

annotations['file'] = annotations['file'].apply(
    lambda x: str(x).split('/')[-1])

annotations['file'] = annotations['file'].apply(
    lambda x: str(x).split('\\')[-1])

annotations['file'] = annotations['file'].apply(
    lambda x: x[:-4] if '.' in x else x)
annotations_double['file'] = annotations_double['file'].apply(
    lambda x: str(x).split('/')[-1])
annotations_double['file'] = annotations_double['file'].apply(
    lambda x: str(x).split('\\')[-1])
annotations_double['file'] = annotations_double['file'].apply(
    lambda x: x[:-4] if '.' in x else x)

annotations['class'] = annotations['class'].apply(lambda x: int(x))
annotations_double['class'] = annotations_double['class'].apply(
    lambda x: int(x))


def get_casel(ann):
    filevis = list(datavis['file'])
    fileinvis = list(datainvis['file'])
    casel = []
    for f in ann['file']:
        if f in filevis:
            casel.append('visible')
        elif f in fileinvis:
            casel.append('invisible')
        else:
            print(f)

    return casel


annotations['case'] = get_casel(annotations)
annotations_double['case'] = get_casel(annotations_double)

annotations.to_csv('annotations_formmatted_with_gaps.csv')
annotations_double.to_csv('annotations_double_formatted_with_gaps.csv')


def fill_gaps(ann, vidstats):
    cols = ['tier', 'beginmm', 'begin', 'endmm', 'end',
            'diffmm', 'diff', 'class', 'file', 'annotator', 'case']
    df = pd.DataFrame(columns=cols)

    # Case annotation not started at 0
    previousrowfile = ''
    for idx, row in ann.iterrows():
        if idx+1 == len(ann.index):
            break
        nextrow = ann.iloc[idx+1]
        duration = float(vidstats[vidstats['file'] == row['file']]['frames'].iloc[0]) / \
            float(vidstats[vidstats['file'] == row['file']]['rate'].iloc[0])
        # case first annotation not started at 0
        if row['begin'] != 0.0 and previousrowfile != row['file']:
            newrow = [row['tier'], 'not_used', 0.0, row['beginmm'], row['begin'], 'not_used',
                      row['begin'], row['class'], row['file'], row['annotator'], row['case']]
            df.loc[len(df)] = newrow
        # case gap in between annotations
        if nextrow['file'] == row['file']:
            if nextrow['begin'] != row['end']:
                diffr = nextrow['begin'] - row['end']
                classn = 4 if diffr > 1 else row['class']
                newrow = [row['tier'], row['endmm'], row['end'], nextrow['beginmm'], nextrow['begin'],
                          'not_used', diffr, classn, row['file'], row['annotator'], row['case']]
                df.loc[len(df)] = newrow
        # case last part not annotated
        elif nextrow['file'] != row['file'] and row['end'] < duration:
            newrow = [row['tier'], row['endmm'], row['end'], 'not_used', duration, 'not_used',
                      duration-row['end'], row['class'], row['file'], row['annotator'], row['case']]
            df.loc[len(df)] = newrow

        previousrowfile = row['file']

    # gap stats
    # print(df['file'].unique(), len(df['file'].unique()), len(df))
    # print(df['diff'].mean())
    # print(df[df['diff']>=1]['file'])

    return pd.concat([ann, df], axis=0)


def get_stats(root, data, cols=['file', 'frames', 'rate']):
    vidstats = pd.DataFrame(columns=['file', 'frames', 'rate'])
    for f in data['file']:
        video = f+'.MP4'
        # print(root+video)
        cap = cv2.VideoCapture(root+video)
        if not cap.isOpened():
            video = f+'.mp4'
            cap = cv2.VideoCapture(root+video)
            if not cap.isOpened():
                raise IOError("Could not read the video file")

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        vidstats.loc[len(vidstats)] = [f, length, fps]

    return vidstats


vidstatsvis = get_stats('../all_vids/visible_with_bounding_boxes/', datavis)
vidstatsinv = get_stats(
    '../all_vids/invisible_with_bounding_boxes/', datainvis)
vidstat = pd.concat([vidstatsvis, vidstatsinv], axis=0)
vidstat.to_csv('vidstat.csv')

annotations = fill_gaps(annotations, vidstat)
annotations_double = fill_gaps(annotations_double, vidstat)
annotations.to_csv('annotations_formatted_gaps_removed.csv')
annotations_double.to_csv('annotations_double_formatted_gaps_removed.csv')


def get_frame_level_data(ann):
    res = pd.DataFrame(
        columns=['file', 'annotator', 'case', 'framenumber', 'class'])

    files = ann[['file', 'annotator']].drop_duplicates()[['file', 'annotator']]
    fs = list(files['file'])
    anns = list(files['annotator'])
    files = list(zip(fs, anns))
    flen=len(files)

    for idx, (f, a) in enumerate(files):
        print('{} out of {}'.format(idx, flen), f,a)
        begins = ann[(ann['file'] == f) & (
            ann['annotator'] == a)][['begin', 'class']]
        # print(begins)
        fcase = list(ann[ann['file'] == f]['case'])[0]
        annotator = a
        begins = begins.reset_index()
        stat = vidstat[vidstat['file'] == f][['frames', 'rate']]
        begin_list = list(begins['begin'])
        class_list = list(begins['class'])
        begins = list(zip(begin_list, class_list))
        begins.sort(key=lambda x: x[0])
        startframes = [(int(b*stat['rate'].iloc[0]), c) for b, c in begins]
        nr_annotations = len(startframes)

        newr = []
        for i in range(nr_annotations):
            start, gazeclass = startframes[i]
            end = startframes[i+1][0] if i < (nr_annotations -
                                              1) else int(stat['frames'].iloc[0])
            newr.append((start, end, int(gazeclass)))

        for start, end, gazeclass in newr:
            for i in range(start, end):
                res.loc[len(res)] = [f, annotator, fcase, i, int(gazeclass)]

        # ensure output is correct
        for i in list(range(int(stat['frames'].iloc[0]))):
            print(i) if i not in list(res['framenumber']) else None

    return res


annotations_frame = get_frame_level_data(annotations)
annotations_frame.to_csv('annotations_frame.csv')

annotations_double_frame = get_frame_level_data(annotations_double)
annotations_double_frame.to_csv('annotations_double_frame.csv')

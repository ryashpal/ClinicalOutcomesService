import pandas as pd


def readData(dataDf, targetStart, targetEnd, windowEnd):

    dataDf.anchor_time = dataDf.anchor_time.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
    dataDf.death_datetime = dataDf.death_datetime.apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))

    if dataDf.empty:
        return None

    dataDf['target'] = (dataDf['death_datetime'] > (dataDf['anchor_time'] + pd.Timedelta(days=targetStart))) & (dataDf['death_datetime'] < (dataDf['anchor_time'] + pd.Timedelta(days=targetEnd)))
    dataDf.fillna({'target': False}, inplace=True)

    dropCols = [
        'person_id',
        'age',
        'gender',
        'ethnicity_WHITE',
        'ethnicity_BLACK',
        'ethnicity_UNKNOWN',
        'ethnicity_OTHER',
        'ethnicity_HISPANIC',
        'ethnicity_ASIAN',
        'ethnicity_UNABLE_TO_OBTAIN',
        'ethnicity_AMERICAN_INDIAN',
        'anchor_time',
        'death_datetime',
        'target',
    ]

    vitalsCols = ['heartrate', 'sysbp', 'diabp', 'meanbp', 'resprate', 'tempc', 'spo2', 'gcseye', 'gcsverbal', 'gcsmotor']
    labsCols = ['chloride_serum', 'creatinine', 'sodium_serum', 'hemoglobin', 'platelet_count', 'urea_nitrogen', 'glucose_serum', 'bicarbonate', 'potassium_serum', 'anion_gap', 'leukocytes_blood_manual', 'hematocrit']

    X = dataDf.drop(dropCols, axis = 1)
    XVitalsMin = dataDf[[vitalCol + '_min' for vitalCol in vitalsCols if vitalCol + '_min' in dataDf.columns]]
    XVitalsMax = dataDf[[vitalCol + '_max' for vitalCol in vitalsCols if vitalCol + '_max' in dataDf.columns]]
    XVitalsAvg = dataDf[[vitalCol + '_avg' for vitalCol in vitalsCols if vitalCol + '_avg' in dataDf.columns]]
    XVitalsSd = dataDf[[vitalCol + '_stddev' for vitalCol in vitalsCols if vitalCol + '_stddev' in dataDf.columns]]
    XVitalsFirst = dataDf[[vitalCol + '_first' for vitalCol in vitalsCols if vitalCol + '_first' in dataDf.columns]]
    XVitalsLast = dataDf[[vitalCol + '_last' for vitalCol in vitalsCols if vitalCol + '_last' in dataDf.columns]]
    XLabsMax = dataDf[[labsCol + '_min' for labsCol in labsCols if labsCol + '_min' in dataDf.columns]]
    XLabsMin = dataDf[[labsCol + '_max' for labsCol in labsCols if labsCol + '_max' in dataDf.columns]]
    XLabsAvg = dataDf[[labsCol + '_avg' for labsCol in labsCols if labsCol + '_avg' in dataDf.columns]]
    XLabsSd = dataDf[[labsCol + '_stddev' for labsCol in labsCols if labsCol + '_stddev' in dataDf.columns]]
    XLabsFirst = dataDf[[labsCol + '_first' for labsCol in labsCols if labsCol + '_first' in dataDf.columns]]
    XLabsLast = dataDf[[labsCol + '_last' for labsCol in labsCols if labsCol + '_last' in dataDf.columns]]
    y = dataDf["target"]

    return X, XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y

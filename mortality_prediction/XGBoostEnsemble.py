import pickle

import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

import logging as log

from utils import DBUtils
from utils import DataUtils


def predict(id, targetStart, targetEnd, windowEnd, savePath, saveFileName):
    log.info('Loading models')
    saveFilePath = savePath + '/' + saveFileName
    allModelsDict = {}
    with open(saveFilePath, 'rb') as f:
        allModelsDict = pickle.load(f)
    log.info('Reading data for the id: ' + id)
    dataDf = DBUtils.getDatamatrixForId(id=id)
    data = DataUtils.readData(dataDf=dataDf, targetStart=targetStart, targetEnd=targetEnd, windowEnd=windowEnd)
    if data:
        X, XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y = data
        XDict = {
            'VitalsMax': XVitalsMax,
            'VitalsMin': XVitalsMin,
            'VitalsAvg': XVitalsAvg,
            'VitalsSd': XVitalsSd,
            'VitalsFirst': XVitalsFirst,
            'VitalsLast': XVitalsLast,
            'LabsMax': XLabsMax,
            'LabsMin': XLabsMin,
            'LabsAvg': XLabsAvg,
            'LabsSd': XLabsSd,
            'LabsFirst': XLabsFirst,
            'LabsLast': XLabsLast,
        }
        standaloneModelsDict = allModelsDict['level_0']
        Xnew = pd.DataFrame()
        for label in standaloneModelsDict.keys():
            for model_name in standaloneModelsDict[label].keys():
                log.info('Performing prediction for the label: ' + label + ', model_name: ' + model_name)
                model = standaloneModelsDict[label][model_name]
                probs = [p for _, p in model.predict_proba(XDict[label])]
                Xnew[label + '_' + model_name] = probs
        probs = [p for _, p in allModelsDict['level_1'].predict_proba(Xnew)][0]
        return {'score': str(probs)}
    else:
        return {'score': None}


def predict_all(targetStart, targetEnd, windowEnd, modelPath, modelFileName, dataPath, dataFileName, saveIntermediate):
    log.info('Loading models')
    modelFilePath = modelPath + '/' + modelFileName
    allModelsDict = {}
    with open(modelFilePath, 'rb') as f:
        allModelsDict = pickle.load(f)
    log.info('Reading data matrix')
    dataDf = DBUtils.getDatamatrix()
    data = DataUtils.readData(dataDf=dataDf, targetStart=targetStart, targetEnd=targetEnd, windowEnd=windowEnd)
    if data:
        X, XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y = data
        XDict = {
            'VitalsMax': XVitalsMax,
            'VitalsMin': XVitalsMin,
            'VitalsAvg': XVitalsAvg,
            'VitalsSd': XVitalsSd,
            'VitalsFirst': XVitalsFirst,
            'VitalsLast': XVitalsLast,
            'LabsMax': XLabsMax,
            'LabsMin': XLabsMin,
            'LabsAvg': XLabsAvg,
            'LabsSd': XLabsSd,
            'LabsFirst': XLabsFirst,
            'LabsLast': XLabsLast,
        }
        standaloneModelsDict = allModelsDict['level_0']
        Xnew = pd.DataFrame()
        for label in standaloneModelsDict.keys():
            for model_name in standaloneModelsDict[label].keys():
                log.info('Performing prediction for the label: ' + label + ', model_name: ' + model_name)
                model = standaloneModelsDict[label][model_name]
                probs = [p for _, p in model.predict_proba(XDict[label])]
                Xnew[label + '_' + model_name] = probs
        log.info('Performing prediction for XGB ensemble')
        finalProbs = [p for _, p in allModelsDict['level_1'].predict_proba(Xnew)]
        if not saveIntermediate:
            Xnew = pd.DataFrame()
        Xnew['XGBensemble'] = finalProbs
        Xnew['y'] = y
        dataFilePath = dataPath + '/' + dataFileName
        Xnew.to_csv(dataFilePath, index=False)
        return 'Successfully created and stored the predictions at: ' + dataFilePath
    else:
        return None


def train(targetStart, targetEnd, windowEnd, savePath, saveFileName):
    dataDf = DBUtils.getDatamatrixForTraining(windowEnd=windowEnd)
    X, XVitalsMax, XVitalsMin, XVitalsAvg, XVitalsSd, XVitalsFirst, XVitalsLast, XLabsMax, XLabsMin, XLabsAvg, XLabsSd, XLabsFirst, XLabsLast, y = DataUtils.readData(dataDf=dataDf, targetStart=targetStart, targetEnd=targetEnd, windowEnd=windowEnd)
    XVitalsMaxTrain, XVitalsMaxTest, XVitalsMinTrain, XVitalsMinTest, XVitalsAvgTrain, XVitalsAvgTest, XVitalsSdTrain, XVitalsSdTest, XVitalsFirstTrain, XVitalsFirstTest, XVitalsLastTrain, XVitalsLastTest, XLabsMaxTrain, XLabsMaxTest, XLabsMinTrain, XLabsMinTest,XLabsAvgTrain, XLabsAvgTest, XLabsSdTrain, XLabsSdTest, XLabsFirstTrain, XLabsFirstTest, XLabsLastTrain, XLabsLastTest, yTrain, yTest = train_test_split(
        XVitalsMax,
        XVitalsMin,
        XVitalsAvg,
        XVitalsSd,
        XVitalsFirst,
        XVitalsLast,
        XLabsMax,
        XLabsMin,
        XLabsAvg,
        XLabsSd,
        XLabsFirst,
        XLabsLast,
        y,
        test_size=0.5,
        random_state=42
        )
    XDict = {
        'VitalsMax': (XVitalsMaxTrain, yTrain, XVitalsMaxTest, yTest),
        'VitalsMin': (XVitalsMinTrain, yTrain, XVitalsMinTest, yTest),
        'VitalsAvg': (XVitalsAvgTrain, yTrain, XVitalsAvgTest, yTest),
        'VitalsSd': (XVitalsSdTrain, yTrain, XVitalsSdTest, yTest),
        'VitalsFirst': (XVitalsFirstTrain, yTrain, XVitalsFirstTest, yTest),
        'VitalsLast': (XVitalsLastTrain, yTrain, XVitalsLastTest, yTest),
        'LabsMax': (XLabsMaxTrain, yTrain, XLabsMaxTest, yTest),
        'LabsMin': (XLabsMinTrain, yTrain, XLabsMinTest, yTest),
        'LabsAvg': (XLabsAvgTrain, yTrain, XLabsAvgTest, yTest),
        'LabsSd': (XLabsSdTrain, yTrain, XLabsSdTest, yTest),
        'LabsFirst': (XLabsFirstTrain, yTrain, XLabsFirstTest, yTest),
        'LabsLast': (XLabsLastTrain, yTrain, XLabsLastTest, yTest),
    }
    log.info('Building standalone models')
    standaloneModelsDict = {}
    for label, (XTrain, yTrain, XTest, yTest) in XDict.items():
        log.info('Models for the label: ' + label)
        standaloneModelsDict[label] = buildStandaloneModels(XTrain, yTrain)
    Xnew = pd.DataFrame()
    for label in standaloneModelsDict.keys():
        for model_name in standaloneModelsDict[label].keys():
            log.info('Performing prediction for the label: ' + label + ', model_name: ' + model_name)
            model = standaloneModelsDict[label][model_name]
            probs = [p for _, p in model.predict_proba(XDict[label][2])]
            auroc = roc_auc_score(XDict[label][3], probs)
            log.info('label: ' + label + ', model: ' + model_name + ' - Model (Testing) AUROC score: ' + str(auroc))
            Xnew[label + '_' + model_name] = probs
    log.info('Performing Hyperparameter optimisation for XGBoost Ensemble model')
    xgbParams = performXgbHyperparameterTuning(Xnew, yTest)
    log.info('Building XGB Ensemble model')
    xgb = XGBClassifier(use_label_encoder=False)
    xgb.set_params(**xgbParams)
    xgb.fit(Xnew, yTest)
    probs = [p for _, p in xgb.predict_proba(Xnew)]
    auroc = roc_auc_score(yTest, probs)
    log.info('XGB Ensemble Model (Training) AUROC score: ' + str(auroc))
    allModelsDict = {'level_1': xgb, 'level_0': standaloneModelsDict}
    saveFilePath = savePath + '/' + saveFileName
    with open(saveFilePath, 'wb') as handle:
        pickle.dump(allModelsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return saveFilePath


def buildStandaloneModels(XTrain, yTrain):
        log.info('Performing Hyperparameter optimisation for XGBoost')
        log.info('Building XGB model')
        xgb = XGBClassifier(use_label_encoder=False)
        xgb.fit(XTrain, yTrain)
        log.info('Performing Hyperparameter optimisation for Logistic Regression')
        log.info('Building LR Model')
        lr = LogisticRegression()
        lr.fit(XTrain, yTrain)
        log.info('Building LGBM Model')
        lgbm = LGBMClassifier(verbose=-1)
        lgbm.fit(XTrain, yTrain)
        log.info('Building MLP Model')
        mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes = (150, 150))
        mlp.fit(XTrain, yTrain)
        return {'xgb': xgb, 'lr': lr, 'lgbm': lgbm, 'mlp': mlp}


def getBestXgbHyperparameter(X, y, tuned_params, parameters):
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    params = {}
    log.info('Hyperparameter optimisation for: ' + str(parameters))
    clf = GridSearchCV(XGBClassifier(use_label_encoder=False, **tuned_params), parameters)
    clf.fit(X, y)
    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]
    return(params)


def performXgbHyperparameterTuning(X, y):
    params = {}
    params.update(getBestXgbHyperparameter(X, y, params, {'max_depth' : range(1,10),'scale_pos_weight': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],}))
    params.update(getBestXgbHyperparameter(X, y, params, {'n_estimators':range(50,250,10)}))
    params.update(getBestXgbHyperparameter(X, y, params, {'min_child_weight':range(1,10)}))
    params.update(getBestXgbHyperparameter(X, y, params, {'gamma':[i/10. for i in range(0,5)]}))
    params.update(getBestXgbHyperparameter(X, y, params, {'subsample':[i/10.0 for i in range(1,10)],'colsample_bytree':[i/10.0 for i in range(1,10)]}))
    params.update(getBestXgbHyperparameter(X, y, params, {'reg_alpha':[0, 1e-5, 1e-3, 0.1, 10]}))
    log.info('params: ' + str(params))
    return params

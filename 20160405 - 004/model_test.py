from libs import store, models
from libs.features import *
from config import *
import xlsxwriter

modelInfos, modelDict = store.loadAllModel()

groups = models.groupModels(modelInfos, modelDict)

data = [['#', 'Learning Rate', 'Learning Rule', 'N_Iter', 'Avg_Distance']]

tmp = store.loadTermData()
termList = {'X': [i[0] for i in tmp], 'y': [int(i[1]) for i in tmp]}
_X = np.asarray([extractFeatureText(term) for term in termList['X']])

for i, igroup in zip(range(len(groups)), groups):

    data.append([
        i,
        igroup['group-info']['learning_rate'],
        igroup['group-info']['learning_rule'],
        igroup['group-info']['n_iter'],
        models.checkModelConvergence(igroup['models'], _X)
    ])

# ================================
workbook = xlsxwriter.Workbook(folder_model + '/' + file_model_result)
store.writeSheet(workbook.add_worksheet('original'), data)
workbook.close()




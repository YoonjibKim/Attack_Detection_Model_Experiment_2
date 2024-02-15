import json
import warnings
import numpy as np
import Constant
from ML_Algorithm.AdaBoostModel import AdaBoostModel
from ML_Algorithm.AgglomerativeClusteringModel import AgglomerativeClusteringModel
from ML_Algorithm.DBSCANModel import DBSCANModel
from ML_Algorithm.DNNModel import DNNModel
from ML_Algorithm.DecisionTreeModel import DecisionTreeModel
from ML_Algorithm.GaussianMixtureModel import GaussianMixtureModel
from ML_Algorithm.GaussianNBModel import GaussianNBModel
from ML_Algorithm.GradientBoostModel import GradientBoostModel
from ML_Algorithm.KMeansModel import KMeansModel
from ML_Algorithm.KNNModel import KNNModel
from ML_Algorithm.LinearRegressionModels import LinearRegressionModels
from ML_Algorithm.LogisticRegressionModel import LogisticRegressionModel
from ML_Algorithm.RandomForestModel import RandomForestModel
from ML_Algorithm.SVMModel import SVMModel


class MLImplementation:
    def __init__(self):
        warnings.filterwarnings("ignore", category=FutureWarning)

        self.__adaBoostModel = AdaBoostModel()
        self.__agglomerativeClusteringModel = AgglomerativeClusteringModel()
        self.__dBSCANModel = DBSCANModel()
        self.__decisionTreeModel = DecisionTreeModel()
        self.__dNNModel = DNNModel()
        self.__gaussianMixtureModel = GaussianMixtureModel()
        self.__gaussianNBModel = GaussianNBModel()
        self.__gradientBoostModel = GradientBoostModel()
        self.__kMeansModel = KMeansModel()
        self.__kNNModel = KNNModel()
        self.__linearRegressionModel = LinearRegressionModels()
        self.__logisticRegressionModel = LogisticRegressionModel()
        self.__randomForestModel = RandomForestModel()
        self.__sVMModel = SVMModel()

        self.__cs_stat_time_delta_dict, self.__gs_stat_time_delta_dict, self.__cs_top_dict, self.__gs_top_dict \
            = self.__load_dataset()

    def __run_other_mls(self, training_feature_array, training_label_array,
                        testing_feature_array, testing_label_array) -> dict:
        adaBoostResult = self.__adaBoostModel.run(training_feature_array, training_label_array,
                                                  testing_feature_array, testing_label_array)
        agglomerativeClusteringResult \
            = self.__agglomerativeClusteringModel.run(testing_feature_array, testing_label_array)
        dBSCANResult = self.__dBSCANModel.run(testing_feature_array, testing_label_array)
        decisionTreeResult = self.__decisionTreeModel.run(training_feature_array, training_label_array,
                                                          testing_feature_array, testing_label_array)
        gaussianMixtureResult = self.__gaussianMixtureModel.run(testing_feature_array, testing_label_array)
        gaussianNBResult = self.__gaussianNBModel.run(training_feature_array, training_label_array,
                                                      testing_feature_array, testing_label_array)
        gradientBoostResult = self.__gradientBoostModel.run(training_feature_array, training_label_array,
                                                            testing_feature_array, testing_label_array)
        kMeansResult = self.__kMeansModel.run(testing_feature_array, testing_label_array)
        kNNResult = self.__kNNModel.run(training_feature_array, training_label_array,
                                        testing_feature_array, testing_label_array)
        linearRegressionResult, ridgeResult, lassoResult, elasticResult \
            = self.__linearRegressionModel.run(training_feature_array, training_label_array,
                                               testing_feature_array, testing_label_array)
        logisticRegressionResult = self.__logisticRegressionModel.run(training_feature_array, training_label_array,
                                                                      testing_feature_array, testing_label_array)
        randomForestResult = self.__randomForestModel.run(training_feature_array, training_label_array,
                                                          testing_feature_array, testing_label_array)
        sVMResult = self.__sVMModel.run(training_feature_array, training_label_array,
                                        testing_feature_array, testing_label_array)

        result_dict = {Constant.MLType.ADA_BOOST: adaBoostResult,
                       Constant.MLType.AGGLOMERATIVE_CLUSTERING: agglomerativeClusteringResult,
                       Constant.MLType.DBSCAN: dBSCANResult,
                       Constant.MLType.DECISION_TREE: decisionTreeResult,
                       Constant.MLType.GAUSSIAN_MIXTURE: gaussianMixtureResult,
                       Constant.MLType.GAUSSIAN_NB: gaussianNBResult,
                       Constant.MLType.GRADIENT_BOOST: gradientBoostResult,
                       Constant.MLType.KMEANS: kMeansResult,
                       Constant.MLType.KNN: kNNResult,
                       Constant.MLType.LINEAR_REGRESSION: linearRegressionResult,
                       Constant.MLType.RIDGE: ridgeResult,
                       Constant.MLType.LASSO: lassoResult,
                       Constant.MLType.ELASTIC: elasticResult,
                       Constant.MLType.LOGISTIC_REGRESSION: logisticRegressionResult,
                       Constant.MLType.RANDOM_FOREST: randomForestResult,
                       Constant.MLType.SVM: sVMResult}

        return result_dict

    @classmethod
    def __load_dataset(cls):
        cs_stat_time_delta_dict = {}
        gs_stat_time_delta_dict = {}
        cs_top_dict = {}
        gs_top_dict = {}

        for scenario, file_name in Constant.SCENARIO_NAME_DICT.items():
            cs_stat_time_delta_file_path = Constant.FileLoad.CS_STAT_TIME_DELTA_DIR_PATH + '/' + file_name + '.json'
            gs_stat_time_delta_file_path = Constant.FileLoad.GS_STAT_TIME_DELTA_DIR_PATH + '/' + file_name + '.json'
            cs_top_file_path = Constant.FileLoad.CS_TOP_DIR_PATH + '/' + file_name + '.json'
            gs_top_file_path = Constant.FileLoad.GS_TOP_DIR_PATH + '/' + file_name + '.json'

            with open(cs_stat_time_delta_file_path, 'r') as f:
                param_cs_stat_time_delta_dict = json.load(f)
            with open(gs_stat_time_delta_file_path, 'r') as f:
                param_gs_stat_time_delta_dict = json.load(f)
            with open(cs_top_file_path, 'r') as f:
                param_cs_top_dict = json.load(f)
            with open(gs_top_file_path, 'r') as f:
                param_gs_top_dict = json.load(f)

            cs_stat_time_delta_dict[scenario] = param_cs_stat_time_delta_dict
            gs_stat_time_delta_dict[scenario] = param_gs_stat_time_delta_dict
            cs_top_dict[scenario] = param_cs_top_dict
            gs_top_dict[scenario] = param_gs_top_dict

        return cs_stat_time_delta_dict, gs_stat_time_delta_dict, cs_top_dict, gs_top_dict

    def __analyze_top_symbol_tournament(self, dataset_dict, dnn_save_path, other_save_path, loss_rate_save_dir):
        dnn_cs_top_result_dict = {}
        other_cs_top_result_dict = {}
        for scenario, category_dict in dataset_dict.items():
            dnn_param_category_dict = {}
            other_param_category_dict = {}
            for category, type_dict in category_dict.items():
                dnn_param_feature_type_dict = {}
                other_param_feature_type_dict = {}
                for feature_type, symbol_dict in type_dict.items():
                    dnn_param_symbol_dict = {}
                    other_param_symbol_dict = {}
                    for symbol, temp_dict in symbol_dict.items():
                        temp_type = scenario + '_' + category + '_' + feature_type + '_' + symbol
                        print(Constant.TOP + ': ' + temp_type)

                        training_label_list = temp_dict[Constant.TRAINING_LABEL]
                        testing_label_list = temp_dict[Constant.TESTING_LABEL]

                        if len(training_label_list) < 2 or len(testing_label_list) < 2:
                            dnn_param_symbol_dict[symbol] = None
                            other_param_feature_type_dict[symbol] = None
                        else:
                            training_feature_array = np.array(temp_dict[Constant.TRAINING_FEATURE])
                            training_label_array = np.array(training_label_list).reshape(-1, 1)
                            testing_feature_array = np.array(temp_dict[Constant.TESTING_FEATURE])
                            testing_label_array = np.array(testing_label_list).reshape(-1, 1)

                            print('DNN')
                            loss_rate_file_name = temp_type + '.png'
                            loss_rate_file_path = loss_rate_save_dir + '/' + loss_rate_file_name
                            dnn_param_result_dict = self.__dNNModel.run(training_feature_array, training_label_array,
                                                                        testing_feature_array, testing_label_array,
                                                                        Constant.Hierarchy.SYMBOL,
                                                                        Constant.DNNParameters.LearningRate.STEP_3,
                                                                        loss_rate_file_path)
                            dnn_param_symbol_dict[symbol] = dnn_param_result_dict
                            # dnn_param_symbol_dict[symbol] = None

                            # print('Other MLs')
                            # other_param_result_dict \
                            #     = self.__run_other_mls(training_feature_array, training_label_array,
                            #                            testing_feature_array, testing_label_array)
                            # other_param_feature_type_dict[symbol] = other_param_result_dict
                            other_param_feature_type_dict[symbol] = None

                    dnn_param_feature_type_dict[feature_type] = dnn_param_symbol_dict
                    other_param_feature_type_dict[feature_type] = other_param_symbol_dict
                dnn_param_category_dict[category] = dnn_param_feature_type_dict
                other_param_category_dict[category] = other_param_feature_type_dict
            dnn_cs_top_result_dict[scenario] = dnn_param_category_dict
            other_cs_top_result_dict[scenario] = other_param_category_dict

        with open(dnn_save_path, 'w') as f:
            json.dump(dnn_cs_top_result_dict, f)
        # with open(other_save_path, 'w') as f:
        #     json.dump(other_cs_top_result_dict, f)

    def __analyze_stat_time_delta_category_tournament(self, dataset_dict, dnn_save_path, other_save_path,
                                                      loss_rate_save_dir):
        dnn_cs_stat_time_delta_result_dict = {}
        other_cs_stat_time_delta_result_dict = {}
        for scenario, category_dict in dataset_dict.items():
            dnn_param_category_dict = {}
            other_param_category_dict = {}
            for category, temp_dict in category_dict.items():
                temp_type = scenario + '_' + category
                print(Constant.STAT_TIME_DELTA + ': ' + temp_type)

                training_feature_array = np.array(temp_dict[Constant.TRAINING_FEATURE])
                training_label_array = np.array(temp_dict[Constant.TRAINING_LABEL]).reshape(-1, 1)
                testing_feature_array = np.array(temp_dict[Constant.TESTING_FEATURE])
                testing_label_array = np.array(temp_dict[Constant.TESTING_LABEL]).reshape(-1, 1)

                print('DNN')
                loss_rate_file_name = temp_type + '.png'
                loss_rate_file_path = loss_rate_save_dir + '/' + loss_rate_file_name
                dnn_param_result_dict \
                    = self.__dNNModel.run(training_feature_array, training_label_array, testing_feature_array,
                                          testing_label_array, Constant.Hierarchy.CATEGORY,
                                          Constant.DNNParameters.LearningRate.STEP_3, loss_rate_file_path)
                dnn_param_category_dict[category] = dnn_param_result_dict
                # dnn_param_category_dict[category] = None

                # print('Other MLs')
                # other_param_category_dict[category] \
                #     = self.__run_other_mls(training_feature_array, training_label_array,
                #                            testing_feature_array, testing_label_array)
                other_param_category_dict[category] = None

            dnn_cs_stat_time_delta_result_dict[scenario] = dnn_param_category_dict
            other_cs_stat_time_delta_result_dict[scenario] = other_param_category_dict

        with open(dnn_save_path, 'w') as f:
            json.dump(dnn_cs_stat_time_delta_result_dict, f)
        # with open(other_save_path, 'w') as f:
        #     json.dump(other_cs_stat_time_delta_result_dict, f)

    def run(self):
        print('Running MLs')

        self.__analyze_top_symbol_tournament(self.__cs_top_dict, Constant.FileSave.DNN_CR_CS_TOP_SYMBOL,
                                             Constant.FileSave.OTHER_CR_CS_TOP_SYMBOL,
                                             Constant.FileSave.DNN_LR_CS_TOP_SYMBOL)
        self.__analyze_top_symbol_tournament(self.__gs_top_dict, Constant.FileSave.DNN_CR_GS_TOP_SYMBOL,
                                             Constant.FileSave.OTHER_CR_GS_TOP_SYMBOL,
                                             Constant.FileSave.DNN_LR_GS_TOP_SYMBOL)

        self.__analyze_stat_time_delta_category_tournament(self.__cs_stat_time_delta_dict,
                                                           Constant.FileSave.DNN_CR_CS_STD_CATEGORY,
                                                           Constant.FileSave.OTHER_CR_CS_STD_CATEGORY,
                                                           Constant.FileSave.DNN_LR_CS_STD_CATEGORY)
        self.__analyze_stat_time_delta_category_tournament(self.__gs_stat_time_delta_dict,
                                                           Constant.FileSave.DNN_CR_GS_STD_CATEGORY,
                                                           Constant.FileSave.OTHER_CR_GS_STD_CATEGORY,
                                                           Constant.FileSave.DNN_LR_GS_STD_CATEGORY)

        print('Running MLs is done.')

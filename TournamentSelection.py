import json
import math
import numpy as np
import Constant
from ML_Implementation.TournamentMatch import TournamentMatch


class TournamentSelection(TournamentMatch):
    def __init__(self):
        super().__init__()

        self.__dnn_cs_std_init_match_result = None
        self.__dnn_cs_top_init_match_result = None
        self.__dnn_gs_std_init_match_result = None
        self.__dnn_gs_top_init_match_result = None

        self.__other_cs_std_init_match_result = None
        self.__other_cs_top_init_match_result = None
        self.__other_gs_std_init_match_result = None
        self.__other_gs_top_init_match_result = None

    def run_match(self):
        # self._run_initial_match()
        self.__load_initial_match_score()

        dnn_cs_top_f1_dict = self.__analyze_dnn_top_init_score(self.__dnn_cs_top_init_match_result)
        dnn_gs_top_f1_dict = self.__analyze_dnn_top_init_score(self.__dnn_gs_top_init_match_result)
        dnn_cs_top_support_dict = self.__analyze_dnn_top_init_score(self.__dnn_cs_top_init_match_result, False)
        dnn_gs_top_support_dict = self.__analyze_dnn_top_init_score(self.__dnn_gs_top_init_match_result, False)

        other_cs_top_f1_dict = self.__analyze_other_top_init_score(self.__other_cs_top_init_match_result)
        other_gs_top_f1_dict = self.__analyze_other_top_init_score(self.__other_gs_top_init_match_result)
        other_cs_top_support_dict = self.__analyze_other_top_init_score(self.__other_cs_top_init_match_result, False)
        other_gs_top_support_dict = self.__analyze_other_top_init_score(self.__other_gs_top_init_match_result, False)

        dnn_cs_std_f1_dict = self.__analyze_dnn_std_init_score(self.__dnn_cs_std_init_match_result)
        dnn_gs_std_f1_dict = self.__analyze_dnn_std_init_score(self.__dnn_gs_std_init_match_result)
        dnn_cs_std_support_dict = self.__analyze_dnn_std_init_score(self.__dnn_cs_std_init_match_result, False)
        dnn_gs_std_support_dict = self.__analyze_dnn_std_init_score(self.__dnn_gs_std_init_match_result, False)

        other_cs_std_f1_dict = self.__analyze_other_std_init_score(self.__other_cs_std_init_match_result)
        other_gs_std_f1_dict = self.__analyze_other_std_init_score(self.__other_gs_std_init_match_result)
        other_cs_std_support_dict = self.__analyze_other_std_init_score(self.__other_cs_std_init_match_result, False)
        other_gs_std_support_dict = self.__analyze_other_std_init_score(self.__other_gs_std_init_match_result, False)

        pass

    @classmethod
    def __analyze_other_top_init_score(cls, top_score_dict, is_f1_score=True) -> dict:
        if is_f1_score:
            sort_type = Constant.F1_SCORE
        else:
            sort_type = Constant.SUPPORT

        param_scenario_dict = {}

        for scenario, category_dict in top_score_dict.items():
            param_f1_list = []
            param_support_list = []
            param_category_dict = {}

            for category, type_dict in category_dict.items():
                param_type_dict_dict = {}

                for type_name, symbol_dict in type_dict.items():
                    if symbol_dict is not None and len(symbol_dict) > 0:
                        param_symbol_dict = {}

                        for feature, temp_dict in symbol_dict.items():
                            param_ml_dict = {}

                            for ml_type, classification_report in temp_dict.items():
                                f1_score = classification_report[Constant.WEIGHTED_AVG][Constant.F1_SCORE]
                                support = classification_report[Constant.WEIGHTED_AVG][Constant.SUPPORT]

                                param_f1_list.append(f1_score)
                                param_support_list.append(support)

                                param_ml_dict[ml_type] = {Constant.F1_SCORE: f1_score, Constant.SUPPORT: support}

                            sorted_data = dict(sorted(param_ml_dict.items(),
                                                      key=lambda item: item[1][sort_type], reverse=True))
                            first_key, first_value = next(iter(sorted_data.items()))
                            first_value[Constant.ML_TYPE] = first_key
                            param_symbol_dict[feature] = first_value

                        sorted_data \
                            = dict(sorted(param_symbol_dict.items(),
                                          key=lambda x: (math.isnan(x[1].get(sort_type, 0)),
                                                         -x[1].get(sort_type, 0))))
                        first_key, first_value = next(iter(sorted_data.items()))
                        first_value[Constant.FEATURE_TYPE] = first_key
                        param_type_dict_dict[type_name] = first_value

                sorted_data = dict(sorted(param_type_dict_dict.items(), key=lambda item: item[1][sort_type],
                                          reverse=True))
                first_key, first_value = next(iter(sorted_data.items()))
                first_value[Constant.TYPE] = first_key
                param_category_dict[category] = first_value

            f1_avg = np.nanmean(param_f1_list)
            support_avg = np.nanmean(param_support_list)

            sorted_data = dict(sorted(param_category_dict.items(), key=lambda item: item[1][sort_type], reverse=True))
            first_key, first_value = next(iter(sorted_data.items()))

            first_value[Constant.F1_AVERAGE] = f1_avg
            first_value[Constant.SUPPORT_AVERAGE] = support_avg

            param_scenario_dict[scenario] = first_value

        return param_scenario_dict

    @classmethod
    def __analyze_dnn_top_init_score(cls, top_score_dict, is_f1_score=True) -> dict:
        if is_f1_score:
            sort_type = Constant.F1_SCORE
        else:
            sort_type = Constant.SUPPORT

        param_scenario_dict = {}

        for scenario, category_dict in top_score_dict.items():
            param_f1_list = []
            param_support_list = []
            param_category_dict = {}

            for category, type_dict in category_dict.items():
                param_type_dict_dict = {}

                for type_name, symbol_dict in type_dict.items():
                    if symbol_dict is not None and len(symbol_dict) > 0:
                        param_symbol_dict = {}

                        for feature, temp_dict in symbol_dict.items():
                            if temp_dict is not None:
                                weighted_avg_dict = temp_dict[Constant.CLASSIFICATION_REPORT][Constant.WEIGHTED_AVG]
                                f1_score = weighted_avg_dict[Constant.F1_SCORE]
                                support = weighted_avg_dict[Constant.SUPPORT]
                                param_f1_list.append(f1_score)
                                param_support_list.append(support)

                                param_symbol_dict[feature] = {Constant.F1_SCORE: f1_score, Constant.SUPPORT: support}

                        if len(param_symbol_dict) > 0:
                            sorted_data \
                                = dict(sorted(param_symbol_dict.items(),
                                              key=lambda x: (math.isnan(x[1].get(sort_type, 0)),
                                                             -x[1].get(sort_type, 0))))
                            first_key, first_value = next(iter(sorted_data.items()))
                            first_value[Constant.FEATURE_TYPE] = first_key
                            param_type_dict_dict[type_name] = first_value

                sorted_data = dict(sorted(param_type_dict_dict.items(), key=lambda item: item[1][sort_type],
                                          reverse=True))
                first_key, first_value = next(iter(sorted_data.items()))
                first_value[Constant.TYPE] = first_key
                param_category_dict[category] = first_value

            f1_avg = np.nanmean(param_f1_list)
            support_avg = np.nanmean(param_support_list)

            sorted_data = dict(sorted(param_category_dict.items(), key=lambda item: item[1][sort_type], reverse=True))
            first_key, first_value = next(iter(sorted_data.items()))

            first_value[Constant.F1_AVERAGE] = f1_avg
            first_value[Constant.SUPPORT_AVERAGE] = support_avg

            param_scenario_dict[scenario] = first_value

        return param_scenario_dict

    @classmethod
    def __analyze_other_std_init_score(cls, std_score_dict, is_f1_score=True) -> dict:
        if is_f1_score:
            sort_type = Constant.F1_SCORE
        else:
            sort_type = Constant.SUPPORT

        param_scenario_dict = {}

        for scenario, category_dict in std_score_dict.items():
            param_category_dict = {}
            f1_list = []
            support_list = []

            for category, ml_dict in category_dict.items():
                param_ml_dict = {}

                for ml_type, temp_dict in ml_dict.items():
                    f1_score = temp_dict[Constant.WEIGHTED_AVG][Constant.F1_SCORE]
                    if not np.isnan(f1_score):
                        f1_list.append(f1_score)
                        support = temp_dict[Constant.WEIGHTED_AVG][Constant.SUPPORT]
                        support_list.append(support)
                        param_ml_dict[ml_type] = {Constant.F1_SCORE: f1_score, Constant.SUPPORT: support}

                if is_f1_score:
                    sorted_element \
                        = dict(sorted(param_ml_dict.items(), key=lambda item: next(iter(item[1].values())),
                                      reverse=True))
                else:
                    sorted_element = dict(sorted(param_ml_dict.items(), key=lambda item: list(item[1].values())[1],
                                                 reverse=True))

                ml_type, temp_dict = next(iter(sorted_element.items()))
                temp_dict[Constant.ML_TYPE] = ml_type

                param_category_dict[category] = temp_dict

            sorted_dict = dict(sorted(param_category_dict.items(), key=lambda item: item[1][sort_type], reverse=True))

            feature_type, temp_dict = next(iter(sorted_dict.items()))

            f1_average = np.mean(f1_list)
            support_average = np.mean(support_list)

            temp_dict[Constant.FEATURE_TYPE] = feature_type
            temp_dict[Constant.F1_AVERAGE] = f1_average
            temp_dict[Constant.SUPPORT_AVERAGE] = support_average

            param_scenario_dict[scenario] = temp_dict

        return param_scenario_dict

    @classmethod
    def __analyze_dnn_std_init_score(cls, std_score_dict, is_f1_score=True) -> dict:
        param_scenario_dict = {}

        for scenario, category_dict in std_score_dict.items():
            param_category_dict = {}

            f1_list = []
            support_list = []

            for category, temp_dict in category_dict.items():
                f1_score = temp_dict[Constant.CLASSIFICATION_REPORT][Constant.WEIGHTED_AVG][Constant.F1_SCORE]
                if not np.isnan(f1_score):
                    f1_list.append(f1_score)
                    support = temp_dict[Constant.CLASSIFICATION_REPORT][Constant.WEIGHTED_AVG][Constant.SUPPORT]
                    support_list.append(support)
                    param_category_dict[category] = {Constant.F1_SCORE: f1_score, Constant.SUPPORT: support}

            if is_f1_score:
                sorted_element = dict(sorted(param_category_dict.items(), key=lambda item: next(iter(item[1].values())),
                                             reverse=True))
            else:
                sorted_element = dict(sorted(param_category_dict.items(), key=lambda item: list(item[1].values())[1],
                                             reverse=True))

            f1_avg = np.mean(f1_list)
            support_avg = np.mean(support_list)

            feature_type, temp_dict = next(iter(sorted_element.items()))

            param_category_dict = {Constant.FEATURE_TYPE: feature_type,
                                   Constant.F1_SCORE: temp_dict[Constant.F1_SCORE],
                                   Constant.SUPPORT: temp_dict[Constant.SUPPORT],
                                   Constant.F1_AVERAGE: f1_avg, Constant.SUPPORT_AVERAGE: support_avg}

            param_scenario_dict[scenario] = param_category_dict

        return param_scenario_dict

    def __load_initial_match_score(self):
        self.__dnn_cs_std_init_match_result \
            = self.__load_file(Constant.CLASSIFICATION_RESULT_FILE_PATH.DNN_CS_STD_INIT_MATCH)
        self.__dnn_cs_top_init_match_result \
            = self.__load_file(Constant.CLASSIFICATION_RESULT_FILE_PATH.DNN_CS_TOP_INIT_MATCH)
        self.__dnn_gs_std_init_match_result \
            = self.__load_file(Constant.CLASSIFICATION_RESULT_FILE_PATH.DNN_GS_STD_INIT_MATCH)
        self.__dnn_gs_top_init_match_result \
            = self.__load_file(Constant.CLASSIFICATION_RESULT_FILE_PATH.DNN_CS_TOP_INIT_MATCH)

        self.__other_cs_std_init_match_result \
            = self.__load_file(Constant.CLASSIFICATION_RESULT_FILE_PATH.OTHER_CS_STD_INIT_MATCH)
        self.__other_cs_top_init_match_result \
            = self.__load_file(Constant.CLASSIFICATION_RESULT_FILE_PATH.OTHER_CS_TOP_INIT_MATCH)
        self.__other_gs_std_init_match_result \
            = self.__load_file(Constant.CLASSIFICATION_RESULT_FILE_PATH.OTHER_GS_STD_INIT_MATCH)
        self.__other_gs_top_init_match_result \
            = self.__load_file(Constant.CLASSIFICATION_RESULT_FILE_PATH.OTHER_GS_TOP_INIT_MATCH)

    @classmethod
    def __load_file(cls, file_path) -> dict:
        try:
            with open(file_path, 'r') as f:
                resultDict = json.load(f)
        except FileNotFoundError:
            print(file_path + ' + is not found.')
            resultDict = None

        return resultDict

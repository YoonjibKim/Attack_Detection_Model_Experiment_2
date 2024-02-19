import json
import Constant
from ML_Implementation.TournamentMatch import TournamentMatch


class TournamentSelection(TournamentMatch):
    def __init__(self):
        super().__init__()
        self.__dnn_dict, self.__other_dict = self.__prepare_result()

        self.__dnn_cs_std_init_match_result = None
        self.__dnn_gs_std_init_match_result = None
        self.__other_cs_std_init_match_result = None
        self.__other_gs_std_init_match_result = None

    def run_match(self):
        self._run_first_match()
        # self.__load_first_match_score()

    def __select_best_symbol_ml(self):
        pass

    def __load_first_match_score(self):
        self.__dnn_cs_std_init_match_result = self.__load_file(Constant.FILE_PATH.DNN_CR_CS_STD_INIT_MATCH)
        self.__dnn_gs_std_init_match_result = self.__load_file(Constant.FILE_PATH.DNN_CR_GS_STD_INIT_MATCH)
        self.__other_cs_std_init_match_result = self.__load_file(Constant.FILE_PATH.OTHER_CR_CS_STD_INIT_MATCH)
        self.__other_gs_std_init_match_result = self.__load_file(Constant.FILE_PATH.OTHER_CR_GS_STD_INIT_MATCH)

    @classmethod
    def __load_file(cls, file_path) -> dict:
        try:
            with open(file_path, 'r') as f:
                resultDict = json.load(f)
        except FileNotFoundError:
            print(file_path + ' + is not found.')
            resultDict = None

        return resultDict

    def __prepare_result(self):
        dnnCsStdInitMatchDict = self.__load_file(Constant.FILE_PATH.DNN_CR_CS_STD_INIT_MATCH)
        dnnCsTopInitMatchDict = self.__load_file(Constant.FILE_PATH.DNN_CR_CS_TOP_INIT_MATCH)
        dnnGsStdInitMatchDict = self.__load_file(Constant.FILE_PATH.DNN_CR_GS_STD_INIT_MATCH)
        dnnGsTopInitMatchDict = self.__load_file(Constant.FILE_PATH.DNN_CR_GS_TOP_INIT_MATCH)

        dnn_cs_top_dict = {Constant.Hierarchy.Initial_Match: dnnCsTopInitMatchDict}
        dnn_cs_std_dict = {Constant.Hierarchy.Initial_Match: dnnCsStdInitMatchDict}
        dnn_cs_dict = {Constant.TOP: dnn_cs_top_dict, Constant.STAT_TIME_DELTA: dnn_cs_std_dict}

        dnn_gs_top_dict = {Constant.Hierarchy.Initial_Match: dnnGsTopInitMatchDict}
        dnn_gs_std_dict = {Constant.Hierarchy.Initial_Match: dnnGsStdInitMatchDict}
        dnn_gs_dict = {Constant.TOP: dnn_gs_top_dict, Constant.STAT_TIME_DELTA: dnn_gs_std_dict}

        dnn_dict = {Constant.CS: dnn_cs_dict, Constant.GS: dnn_gs_dict}

        otherCsStdInitMatchDict = self.__load_file(Constant.FILE_PATH.OTHER_CR_CS_STD_INIT_MATCH)
        otherCsTopInitMatchDict = self.__load_file(Constant.FILE_PATH.OTHER_CR_CS_TOP_INIT_MATCH)
        otherGsStdInitMatchDict = self.__load_file(Constant.FILE_PATH.OTHER_CR_GS_STD_INIT_MATCH)
        otherGsTopInitMatchDict = self.__load_file(Constant.FILE_PATH.OTHER_CR_GS_TOP_INIT_MATCH)

        other_cs_top_dict = {Constant.Hierarchy.Initial_Match: otherCsStdInitMatchDict}
        other_cs_std_dict = {Constant.Hierarchy.Initial_Match: otherCsTopInitMatchDict}
        other_cs_dict = {Constant.TOP: other_cs_top_dict, Constant.STAT_TIME_DELTA: other_cs_std_dict}

        other_gs_top_dict = {Constant.Hierarchy.Initial_Match: otherGsTopInitMatchDict}
        other_gs_std_dict = {Constant.Hierarchy.Initial_Match: otherGsStdInitMatchDict}
        other_gs_dict = {Constant.TOP: other_gs_top_dict, Constant.STAT_TIME_DELTA: other_gs_std_dict}

        other_dict = {Constant.CS: other_cs_dict, Constant.GS: other_gs_dict}

        return dnn_dict, other_dict

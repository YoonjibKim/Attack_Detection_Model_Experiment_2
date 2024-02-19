import json
import Constant
from ML_Implementation.TournamentMatch import TournamentMatch


class TournamentSelection(TournamentMatch):
    def __init__(self):
        super().__init__()

        self.__dnn_cs_std_init_match_result = None
        self.__dnn_gs_std_init_match_result = None
        self.__other_cs_std_init_match_result = None
        self.__other_gs_std_init_match_result = None

    def run_match(self):
        self._run_initial_match()
        self.__load_initial_match_score()

    def __select_best_initial_match_result(self):
        pass

    def __load_initial_match_score(self):
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

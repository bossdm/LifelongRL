import sys
if sys.version_info[0] == 2:
    import ConfigParser
else:
    import configparser as ConfigParser


import copy
from Configs.InstructionSets import *
# todo: SSA_WM_params: eval; load instruction_sets properly

def parse_booleans(config,sec_name,params):
    print(config._sections)
    print(config._sections[sec_name])

    for key in config._sections[sec_name]:
        if key == "__name__":
            continue
        fl = config.getboolean(sec_name, key)
        params.update({str(key): fl})


def parse_ints(config,sec_name,params):
    print(config._sections)
    print(config._sections[sec_name])
    for key in config._sections[sec_name]:
        if key == "__name__":
            continue
        fl = config.getint(sec_name,key)
        params.update({str(key): fl})
def parse_floats(config,sec_name,params):
    print(config._sections)
    print(config._sections[sec_name])
    for key in config._sections[sec_name]:
        if key == "__name__":
            continue
        fl = config.getfloat(sec_name,key)
        params.update({str(key): fl})
def parse_strings(outfile,config,sec_name,defaultparams):
    print(config._sections)
    print(config._sections[sec_name])
    for key in config._sections[sec_name]:
        if key == "__name__":
            continue
        fl = config.get(sec_name,key)
        print(key)
        print(outfile)
        defaultparams.update({str(key): fl})
def parse_stringlists(config,sec_name,params):
    import json
    # get a list of strings
    for key in config._sections[sec_name]:
        if key == "__name__":
            continue
        list = json.loads(config.get(sec_name, key))
        params.update({str(key): list})

def parse_config_file(outfile,filename,params):
    configr = ConfigParser.ConfigParser()
    config_file = filename
    if not configr.read([config_file]):
        msg='cannot load config ' + str(config_file)
        raise IOError(msg)
    configr.BOOLEAN_STATES = {'true': True, 'false': False}
    parse_booleans(configr,'booleans',params)
    parse_ints(configr,'ints',params)
    parse_floats(configr,'floats',params)
    outfile=parse_strings(outfile,configr,'strings',params)
    parse_stringlists(configr,'stringlists',params)
    return outfile
def setSSA_WM_Params(SSAWMParams):
    #SSAWMParams['predictiveSelfMod']=params['predictiveSelfMod']
    # if SSAWMParams['eval']:
    #     SSAWMParams['internal_actionsSSA'].update(searchPset)
    # elif SSAWMParams['predictiveSelfMod']:
    #     SSAWMParams['internal_actionsSSA'].update(predictiveSelfMod)
    # else:
    #
    SSAWMParams['internal_actionsSSA'].update(incPset)

def setIS_NEAT_params(params,SSANeat_Params,networktypeMap):
    SSANeat_Params['instruction_sets']=[networktypeMap[instr] for instr in params['network_types']]



def getIS_NEAT_configs(config,SSANeat_Params):
    configs = []
    sets=SSANeat_Params['instruction_sets']

    # if len(sets) > 1:
    #     SSANeat_Params['use_setnet'] = True
    for set in sets:
        config['instruction_set'] = set
        configs.append(copy.copy(config))
    return configs

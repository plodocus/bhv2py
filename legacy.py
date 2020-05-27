import struct
import numpy as np
import pandas as pd


class reader:
    def __init__(self, endianness):
        self.endianness = endianness
        self.ml_dtypes = {
                'uint32': (4, 'I'),
                'uchar': (1, 'B'),
                'double': (8, 'd'),
                'uint16': (2, 'H'),
                'int16': (2, 'h'),
                'uint8': (1, 'B'),
                'single': (4, 'f'),
                }

    def read(self, f, dtype, n=1):
        this_dtype = self.ml_dtypes[dtype]
        buf = f.read(this_dtype[0]*n)

        if dtype == 'uchar':
            return buf.decode('utf-8').strip()
        else:
            if self.endianness == 'little':
                fmt = '<'
            elif self.endianness == 'big':
                fmt = '>'

            return struct.unpack(fmt+str(n)+this_dtype[1], buf)[0]


def read_bhv(filename):
    bhv = {}
    r = reader('little')
    with open(filename, 'rb') as f:
        bhv['MagicNumber'] = r.read(f, 'uint32', 1)
        bhv['FileHeader'] = r.read(f, 'uchar', 64)
        bhv['FileVersion'] = r.read(f, 'double', 1)
        bhvver = bhv['FileVersion']
        bhv['StartTime'] = r.read(f, 'uchar', 32)
        bhv['ExperimentName'] = r.read(f, 'uchar', 128)
        if bhvver > 1.5:
            bhv['Investigator'] = r.read(f, 'uchar', 128)
        bhv['SubjectName'] = r.read(f, 'uchar', 128)
        if bhvver > 2.1:
            bhv['ComputerName'] = r.read(f, 'uchar', 128)
        bhv['ConditionsFile'] = r.read(f, 'uchar', 128)
        num_cnds = r.read(f, 'uint16', 1)
        obj_per_cnd = r.read(f, 'uint16', 1)
        task_obj = r.read(f, 'uchar', 64*num_cnds*obj_per_cnd)
        task_obj = np.array([task_obj[(i*64):(i+1)*64].strip() for i in range(num_cnds*obj_per_cnd)])
        task_obj = task_obj.reshape(obj_per_cnd, num_cnds).T
        bhv['NumConds'] = num_cnds
        bhv['ObjectsPerCond'] = obj_per_cnd
        bhv['TaskObject'] = task_obj
        if bhvver > 2.65:
            bhv['TimingFileByCond'] = r.read(f, 'uchar', bhv['NumConds']*128)
            if bhvver > 2.71:
                maxblocks = r.read(f, 'uint8', 1)
                bhv['BlockByCond'] = r.read(f, 'uint8', bhv['NumConds']*maxblocks)
            else:
                bhv['BlockByCond'] = r.read(f, 'uint8', bhv['NumConds'])
            bhv['InfoByCond'] = r.read(f, 'uchar', bhv['NumConds']*128)
        num_timing_files = r.read(f, 'uint8', 1)
        bhv['TimingFiles'] = r.read(f, 'uchar', 128*num_timing_files)
        bhv['ErrorLogic'] = r.read(f, 'uchar', 64)
        bhv['BlockLogic'] = r.read(f, 'uchar', 64)
        bhv['CondLogic'] = r.read(f, 'uchar', 64)
        bhv['BlockSelectFunction'] = r.read(f, 'uchar', 64)
        bhv['CondSelectFunction'] = r.read(f, 'uchar', 64)
        if bhvver > 2.0:
            bhv['VideoRefreshRate'] = r.read(f, 'double', 1)
            if bhvver > 3.0:
                bhv['ActualVideoRefreshRate'] = r.read(f, 'double', 1)
            bhv['VideoBufferPages'] = r.read(f, 'uint16', 1)
        bhv['ScreenXresolution'] = r.read(f, 'uint16', 1)
        bhv['ScreenYresolution'] = r.read(f, 'uint16', 1)
        bhv['ViewingDistance'] = r.read(f, 'double', 1)
        bhv['PixelsPerDegree'] = r.read(f, 'double', 1)
        if bhvver > 2.01:
            bhv['AnalogInputType'] = r.read(f, 'uchar', 32)
            bhv['AnalogInputFrequency'] = r.read(f, 'double', 1)
        if bhvver > 2.0:
            bhv['AnalogInputDuplication'] = r.read(f, 'uchar', 32)

        bhv['EyeSignalCalibrationMethod'] = r.read(f, 'uchar', 32)
        tmatrix = r.read(f, 'uint8', 1)
        if bhvver > 4.0:
            tmatrix = 1 + tmatrix * 2
        if tmatrix == 1:
            bhv['EyeTransform'] = None
        elif tmatrix == 2:
            bhv['EyeTransform'] = {
                    'origin': r.read(f, 'double', 2),
                    'gain': r.read(f, 'double', 2),
                    }
        elif tmatrix == 3:
            bhv['EyeTransform'] = {
                    'ndims_in': r.read(f, 'uint16', 1),
                    'ndims_out': r.read(f, 'uint16', 1),
                    'forward_fcn': r.read(f, 'uchar', 64), # *char in ML code
                    'inverse_fcn': r.read(f, 'uchar', 64), # *char in ML code
                    }
            tsize = r.read(f, 'uint16', 1)
            tsqrt = tsize**(1/2)
            bhv['EyeTransform']['tdata'] = {
                    'T': r.read(f, 'double', tsize), # reshape this
                    'Tinv': r.read(f, 'double', tsize), # reshape this
                    }

        bhv['JoystickCalibrationMethod'] = r.read(f, 'uchar', 32)
        tmatrix = r.read(f, 'uint8', 1)
        if bhvver > 4.0:
            tmatrix = 1 + tmatrix * 2
        if tmatrix == 1:
            bhv['JoyTransform'] = None
        elif tmatrix == 2:
            bhv['JoyTransform'] = {
                    'origin': r.read(f, 'double', 2),
                    'gain': r.read(f, 'double', 2),
                    }
        elif tmatrix == 3:
            bhv['JoyTransform'] = {
                    'ndims_in': r.read(f, 'uint16', 1),
                    'ndims_out': r.read(f, 'uint16', 1),
                    'forward_fcn': r.read(f, 'uchar', 64), # *char in ML code
                    'inverse_fcn': r.read(f, 'uchar', 64), # *char in ML code
                    }
            tsize = r.read(f, 'uint16', 1)
            tsqrt = tsize**(1/2)
            bhv['JoyTransform']['tdata'] = {
                    'T': r.read(f, 'double', tsize), # reshape this
                    'Tinv': r.read(f, 'double', tsize), # reshape this
                    }

        bhv['PhotoDiodePosition'] = r.read(f, 'uchar', 12)

        if bhvver > 1.9:
            bhv['ScreenBackgroundColor'] = r.read(f, 'double', 3)
            bhv['EyeTraceColor'] = r.read(f, 'double', 3)
            bhv['JoyTraceColor'] = r.read(f, 'double', 3)

        bhv['Stimuli'] = {'NumPics': r.read(f, 'uint16', 1)}
        PIC = []
        for i in range(bhv['Stimuli']['NumPics']):
            PIC.append({'Name': r.read(f, 'uchar', 128)})
        for pic in PIC:
            pic['Size'] = r.read(f, 'uint16', 3)
        for pic in PIC:
            sz = pic['Size']
            sz = sz[0]*sz[1]*sz[2]
            pic['Data'] = r.read(f, 'uint8', sz)
        bhv['Stimuli']['PIC'] = PIC

        if bhvver > 2.5:
            MOV = []
            bhv['Stimuli']['NumMovs'] = r.read(f, 'uint16', 1)
            for i in range(bhv['Stimuli']['NumMovs']):
                MOV.append({'Name': r.read(f, 'uchar', 128)})
            for mov in MOV:
                sz = r.read(f, 'uint16', 4)
                mov['Size'] = sz[:3]
                mov['NumFrames'] = sz[3]
            for mov in MOV:
                sz = mov['Size']
                sz = sz[0]*sz[1]*sz[2]
                data = []
                for fr in mov['NumFrames']:
                    data.append(r.read(f, 'uint8', sz))
                mov['Data'] = data
            bhv['Stimuli']['MOV'] = MOV

        padding = r.read(f, 'uint8', 1024)
        bhv['NumTrials'] = r.read(f, 'uint16', 1)
        n_trl = bhv['NumTrials']

        trialnumber = np.zeros(n_trl, dtype=np.int)
        blocknumber = np.zeros(n_trl, dtype=np.int)
        condnumber = np.zeros(n_trl, dtype=np.int)
        joyrotationdeg = np.zeros(n_trl, dtype=np.int)
        trialerror = np.zeros(n_trl, dtype=np.int)
        mincyclerate = np.zeros(n_trl, dtype=np.int)
        cyclerate = np.zeros(n_trl, dtype=np.int)
        numcodes = np.zeros(n_trl, dtype=np.int)
        rt = np.zeros(n_trl, dtype=np.int)
        codes = {}
        analogdata = {}
        objectstatusrecord = {}
        rewardrecord = {}
        uservars = {}


        if bhvver > 4:
            abs_trial_start_time = np.zeros(n_trl)
        else:
            abs_trial_start_time = np.empty(n_trl, dtype='datetime64[ms]')
            for tr in range(bhv['NumTrials']):
                abs_trial_start_time[tr] = np.datetime64('nat','ms')

        trial_datetime = np.zeros(n_trl)
        for tr in range(bhv['NumTrials']):
            trialnumber[tr] = r.read(f, 'uint16', 1)
            if bhvver > 4:
                abs_trial_start_time[tr] = r.read(f, 'double', 1)
                numc = r.read(f, 'uint8', 1)
                trial_datetime[tr] = r.read(f, 'double', numc)
            else:
                numc = r.read(f, 'uint8', 1)
                t = [r.read(f, 'double', 1) for i in range(numc)]
                dattime_str = f'{t[0]:.0f}-{t[1]:02.0f}-{t[2]:02.0f}T{t[3]:02.0f}:{t[4]:02.0f}:{t[5]:06.3f}'
                abs_trial_start_time[tr] = np.datetime64(dattime_str,'ms')
            blocknumber[tr] = r.read(f, 'uint16')
            # block index missing
            condnumber[tr] = r.read(f, 'uint16')
            if bhvver >= 3.2:
                joyrotationdeg[tr] = r.read(f, 'uint16')
            trialerror[tr] = r.read(f, 'uint16')
            if bhvver >= 2.05:
                mincyclerate[tr] = r.read(f, 'uint16')
                if bhvver >= 4.0:
                    cyclerate[tr] = r.read(f, 'uint16')
                elif bhvver >= 2.72:
                    if mincyclerate[tr] > 0:
                        cyclerate[tr] = r.read(f, 'uint16')
                    else:
                        cyclerate[tr] = 0

            numcodes[tr] = r.read(f, 'uint16')
            codenumbers = [r.read(f, 'uint16') for i in range(numcodes[tr])]
            if bhvver >= 4.2:
                codetimes = [r.read(f, 'double') for i in range(numcodes[tr])]
            elif bhvver >= 3.0:
                codetimes = [r.read(f, 'uint32') for i in range(numcodes[tr])]
            else:
                codetimes = [r.read(f, 'uint16') for i in range(numcodes[tr])]
            codes[tr] = pd.DataFrame({'Numbers': codenumbers, 'Times': codetimes})

            ai = {}
            if bhvver >= 1.5:
                num_x_eye_points = r.read(f, 'uint32')
                if num_x_eye_points > 0:
                    if bhvver > 1.6:
                        xeye = [r.read(f, 'single') for i in range(num_x_eye_points)]
                    else:
                        xeye = [r.read(f, 'double') for i in range(num_x_eye_points)]

                num_y_eye_points = r.read(f, 'uint32')
                if num_y_eye_points > 0:
                    if bhvver > 1.6:
                        yeye = [r.read(f, 'single') for i in range(num_y_eye_points)]
                    else:
                        yeye = [r.read(f, 'double') for i in range(num_y_eye_points)]

                if num_x_eye_points == num_y_eye_points and num_x_eye_points > 0:
                    ai['EyeSignal'] = [xeye,yeye]
                elif num_x_eye_points > num_y_eye_points:
                    ai['EyeSignal'] = xeye


            if bhvver >= 1.5:
                num_x_joy_points = r.read(f, 'uint32')
                if num_x_joy_points > 0:
                    if bhvver > 1.6:
                        xjoy = [r.read(f, 'single') for i in range(num_x_joy_points)]
                    else:
                        xjoy = [r.read(f, 'double') for i in range(num_x_joy_points)]

                num_y_joy_points = r.read(f, 'uint32')
                if num_y_joy_points > 0:
                    if bhvver > 1.6:
                        yjoy = [r.read(f, 'single') for i in range(num_y_joy_points)]
                    else:
                        yjoy = [r.read(f, 'double') for i in range(num_y_joy_points)]

                if num_x_joy_points == num_y_joy_points and num_x_joy_points > 0:
                    ai['JoySignal'] = [xjoy,yjoy]
                elif num_x_joy_points > num_y_joy_points:
                    ai['JoySignal'] = xjoy


            # skipped Eddie Ryklin compatibility

            if bhvver >= 4.0:
                numMouseData = r.read(f, 'uint32')
                ai['MouseData'] = [r.read(f, 'single') for i in range(4*numMouseData)]
                # skipped reshaping

            ai['General'] = {}
            if bhvver >= 2.5:
                nGen = 9
                if bhvver >= 4.0:
                    nGen = 16
                for i in range(nGen):
                    gname = f'Gen{i+1}'
                    num_gen_points = r.read(f, 'uint32')
                    gen = [r.read(f, 'single') for i in range(num_gen_points)]
                    ai['General'][gname] = gen

            if bhvver >= 1.8:
                num_photo_diode_points = r.read(f, 'uint32')
                if num_photo_diode_points > 0:
                    phdio = [r.read(f, 'single') for i in range(num_photo_diode_points)]
                else:
                    phdio = []
                ai['PhotoDiode'] = phdio

            analogdata[tr] = ai


            if bhvver >= 4.2:
                rt[tr] = r.read(f, 'double')
            else:
                rt[tr] = r.read(f, 'int16')


            if bhvver >= 1.9:
                if bhvver >= 4.0:
                    raise NotImplementedError
                elif bhvver >= 2.00:
                    numstat = r.read(f, 'uint32')
                    data = []
                    status = []
                    time = []
                    for i in range(numstat):
                        numbits = r.read(f, 'uint32')
                        status.append([r.read(f, 'uint8') for i in range(numbits)])
                        time.append(r.read(f, 'uint32'))
                        data_this = []
                        if np.any(np.array(status[-1])>1):
                            numfields = r.read(f, 'uint8')
                            for fnum in range(numfields):
                                datacount = r.read(f, 'uint32')
                                print(datacount)
                                data_this.append([r.read(f, 'double') for i in range(datacount)])
                        data.append(data_this)
                    objectstatusrecord[tr] = {
                            'Status': np.array(status, dtype=np.int),
                            'Time': np.array(time),
                            'Data': data,
                            }
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            # rewards
            if bhvver >= 1.95:
                numreward = r.read(f, 'uint32')
                if bhvver >= 4.2:
                    raise NotImplementedError
                else:
                    on = [r.read(f, 'uint32') for i in range(numreward)]
                    off = [r.read(f, 'uint32') for i in range(numreward)]
                    rewardrecord[tr] = {
                            'RewardOnTime': on,
                            'RewardOffTime': off,
                            }
            else:
                raise NotImplementedError


            # user vars
            if bhvver >= 2.7:
                num_user_vars = r.read(f, 'uint8')
                if num_user_vars > 0:
                    uservars[tr] = {}
                for i in range(num_user_vars):
                    varname = r.read(f, 'uchar', 32).strip()
                    vartype = r.read(f, 'uchar').strip()
                    if vartype == 'd':
                        varlen = r.read(f, 'uint8')
                        varval = [r.read(f, 'double') for i in range(varlen)]
                    elif vartype == 'c':
                        varval = r.read(f, 'uchar', 128).strip()
                    else:
                        varval = []
                    uservars[tr][varname] = varval

        # skipped block order

        # this data is discarded
        n_beh_codes_used = r.read(f, 'uint16')
        bhv['CodesUsed'] = {
                'Numbers': [r.read(f, 'uint16') for i in range(n_beh_codes_used)],
                'Name': [r.read(f, 'uchar', 64).strip() for i in range(n_beh_codes_used)],
                }

        if bhvver >= 2.05:
            numf = r.read(f, 'uint16')
            VarChanges = {}
            for i in range(numf):
                fn = r.read(f, 'uchar', 64).strip()
                n = r.read(f, 'uint16')
                VarChanges[fn] = {
                        'Trial': [r.read(f, 'uint16') for i in range(n)],
                        'Value': [r.read(f, 'double') for i in range(n)],
                        }
            bhv['VariableChanges'] = VarChanges

        bhv['FinishTime'] = r.read(f, 'uchar', 32)

    trials_dict = {
            'TrialNumber': trialnumber,
            'AbsoluteTrialStartTime': abs_trial_start_time,
            'TrialDateTime': trial_datetime,
            'BlockNumber': blocknumber,
            'ConditionNumber': condnumber,
            'JoyRotationDeg': joyrotationdeg,
            'TrialError': trialerror,
            'MinCycleRate': mincyclerate,
            'CycleRate': cyclerate,
            'NumCodes': numcodes,
            'Codes': list(codes.values()),
            'ReactionTime': rt,
            'ObjectStatusRecord': list(objectstatusrecord.values()),
            'AnalogData': list(analogdata.values()),
            }
    bhv['Trials'] = pd.DataFrame.from_dict(trials_dict)

    bhv['StartTime'] = pd.to_datetime(bhv['StartTime'], format='%d-%b-%Y %H:%M:%S')

    return bhv

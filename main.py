import mat73
import numpy as np
import math
import plotly.graph_objects as go
from scipy.io import savemat


# Visualise the local spatial neighbourhood about the EOI
def plot_neighbourhood(Neighbours, OptFlowEst, Eoi):
    tOptFlowEstdeltaP = list(zip(*OptFlowEst['deltaP']))
    Neighbours['x'] = Neighbours['deltaX'] + Eoi['x']
    Neighbours['y'] = Neighbours['deltaY'] + Eoi['y']
    mny = min(min(Neighbours['y'][Neighbours['isSamePolarity']]), Eoi['y'],
              min(np.array(tOptFlowEstdeltaP[1]) + Eoi['y']))
    mxy = max(max(Neighbours['y'][Neighbours['isSamePolarity']]), Eoi['y'],
              max(np.array(tOptFlowEstdeltaP[1]) + Eoi['y']))
    mnx = min(min(Neighbours['x'][Neighbours['isSamePolarity']]), Eoi['x'],
              min(np.array(tOptFlowEstdeltaP[0]) + Eoi['x']))
    mxx = max(max(Neighbours['x'][Neighbours['isSamePolarity']]), Eoi['x'],
              max(np.array(tOptFlowEstdeltaP[0]) + Eoi['x']))
    layout = go.Layout(title_text='Local spatial neighbourhood about the EOI',
                       scene=dict(xaxis=dict(title='X', range=[mnx - 1, mxx + 1]),
                                  yaxis=dict(title='Y', range=[mny - 1, mxy + 1]),
                                  zaxis=dict(title='T', range=[min(Neighbours['lastTs'].flatten()),
                                                               max(np.array(OptFlowEst['deltaTs']) + Eoi[
                                                                   'ts'])]))
                       )
    fig = go.Figure(data=[go.Surface(x=Neighbours['x'], y=Neighbours['y'], z=Neighbours['lastTs'])],
                    layout=layout)
    fig.add_scatter3d(x=Neighbours['x'][Neighbours['isSamePolarity']],
                      y=Neighbours['y'][Neighbours['isSamePolarity']],
                      z=Neighbours['lastTs'][Neighbours['isSamePolarity']])

    z = np.array(OptFlowEst['deltaTs']) + Eoi['ts']
    fig.add_scatter3d(x=np.array(tOptFlowEstdeltaP[1]) + Eoi['x'],
                      y=np.array(tOptFlowEstdeltaP[0]) + Eoi['y'],
                      z=z.flatten()
                      )
    fig.add_scatter3d(x=[Eoi['x']], y=[Eoi['y']], z=[Eoi['ts']])
    # legend
    fig.update_layout(showlegend=False)

    # x axis
    fig.update_xaxes(visible=False)

    # y axis
    fig.update_yaxes(visible=False)

    fig.show()
    input("Press Enter to continue...")


if __name__ == '__main__':
    visualise_spatial_neighbourhood = True
    path = 'sequences/'
    sequence_names = ['stripes.mat', 'rotating_bar.mat', 'slider_hdr_far.mat']
    data_dict = mat73.loadmat(path + sequence_names[1])

    # load saved sequence
    TD = data_dict['TD']

    # event replay simulation parameters
    SimParams = {'FIRST_EVENT_IDX': 0, 'LAST_EVENT_IDX': 180000, 'OPT_FLOW_EST_STEPS': 170000}

    # SOFEA parameters
    OptFlowEstParams = {'L': 7, 'T_RF': 40000, 'N_NB': 16, 'E': 11000, 'N_SP': 15}

    # derived parameters
    screen_size = (1080, 1920)
    NewEvent = dict()
    Neighbours = dict()
    Neighbours['length'] = OptFlowEstParams['L']
    Neighbours['halfLength'] = math.floor((OptFlowEstParams['L'] - 1) / 2)
    Neighbours['count'] = OptFlowEstParams['L'] ** 2
    Neighbours['centerIdx'] = Neighbours['halfLength']
    Neighbours['relativeX'], Neighbours['relativeY'] = np.meshgrid(
        np.linspace(0, OptFlowEstParams['L'] - 1, OptFlowEstParams['L']),
        np.linspace(0, OptFlowEstParams['L'] - 1, OptFlowEstParams['L']))
    Neighbours['relativeP'] = np.array(
        [list(Neighbours['relativeX'].flatten()), list(Neighbours['relativeY'].flatten())]).T.tolist()
    Neighbours['deltaX'] = Neighbours['relativeX'] - Neighbours['centerIdx']
    Neighbours['deltaY'] = Neighbours['relativeY'] - Neighbours['centerIdx']
    Neighbours['deltaP'] = np.array([Neighbours['deltaX'].flatten(), Neighbours['deltaY'].flatten()]).T.tolist()

    FrameRes = {'x': max(TD['x']), 'y': max(TD['y'])}
    FrameRes['size'] = (int(FrameRes['y']), int(FrameRes['x']))

    prev_gray = np.zeros(FrameRes['size'])

    # initialisations
    Events = {'lastP': np.zeros(FrameRes['size']), 'lastTs': np.zeros(FrameRes['size'])}

    OptFlowEval = {'velocities': np.zeros((SimParams['OPT_FLOW_EST_STEPS'], 2)),
                   'x': np.zeros((SimParams['OPT_FLOW_EST_STEPS'], 1)),
                   'y': np.zeros((SimParams['OPT_FLOW_EST_STEPS'], 1)),
                   'p': np.zeros((SimParams['OPT_FLOW_EST_STEPS'], 1)),
                   'ts': np.zeros((SimParams['OPT_FLOW_EST_STEPS'], 1)), 'eventIdx': 0}
    # event-by-event process loop
    for eventIdx in range(SimParams['FIRST_EVENT_IDX'] - 1, SimParams['LAST_EVENT_IDX']):
        Eoi = {'x': int(TD['x'][eventIdx - 1]) - 1, 'y': int(TD['y'][eventIdx - 1]) - 1,
               'p': int(TD['p'][eventIdx - 1]), 'ts': int(TD['ts'][eventIdx - 1])}
        # constant refractory period filtering
        if (Eoi['ts'] - Events['lastTs'][Eoi['y']][Eoi['x']]) <= OptFlowEstParams['T_RF']:
            continue

        # record all 'Eoi.ts' & 'Eoi.p' except for refractory filtered ones
        Events['lastTs'][Eoi['y']][Eoi['x']] = Eoi['ts']
        Events['lastP'][Eoi['y']][Eoi['x']] = Eoi['p']

        if eventIdx <= SimParams['LAST_EVENT_IDX'] - SimParams['OPT_FLOW_EST_STEPS']:
            continue
        # estimate optical flow of the EOI
        # greedy selection of OptFlowEstParams['N_NB'] associated spatial neighbours of the EOI
        # only process EOI with 'OptFlowEstParams['L'] x OptFlowEstParams['L']' neighbours
        NbPosRange = {'xl': Eoi['x'] - Neighbours['halfLength'], 'xh': Eoi['x'] + Neighbours['halfLength'],
                      'yl': Eoi['y'] - Neighbours['halfLength'], 'yh': Eoi['y'] + Neighbours['halfLength']}
        if NbPosRange['xl'] < 0 or NbPosRange['xh'] > FrameRes['x'] - 1 or NbPosRange['yl'] < 0 or NbPosRange['yh'] > \
                FrameRes['y'] - 1:
            continue
        # extract spatial neighbours of the EOI
        Neighbours['lastTs'] = np.array(
            Events['lastTs'][NbPosRange['yl']: NbPosRange['yh'] + 1, NbPosRange['xl']: NbPosRange['xh'] + 1])
        Neighbours['lastP'] = np.array(
            Events['lastP'][NbPosRange['yl']: NbPosRange['yh'] + 1, NbPosRange['xl']: NbPosRange['xh'] + 1])
        Neighbours['lastTs'] = np.pad(Neighbours['lastTs'],
                                      [(0, NbPosRange['yh'] + 1 - NbPosRange['yl'] - Neighbours['lastTs'].shape[0]),
                                       (0, NbPosRange['xh'] + 1 - NbPosRange['xl'] - Neighbours['lastTs'].shape[1])])
        Neighbours['lastP'] = np.pad(Neighbours['lastP'],
                                     [(0, NbPosRange['yh'] + 1 - NbPosRange['yl'] - Neighbours['lastP'].shape[0]),
                                      (0, NbPosRange['xh'] + 1 - NbPosRange['xl'] - Neighbours['lastP'].shape[1])])
        Neighbours['isSamePolarity'] = np.array(Eoi['p'] == Neighbours['lastP'])

        # sort neighbouring events based on their timestamps
        Neighbours['lastTs'][Neighbours['centerIdx']][Neighbours['centerIdx']] = \
            Neighbours['lastTs'][Neighbours['centerIdx']][Neighbours['centerIdx']] + 1
        # done to ensure the EOI has the largest timestamp
        Neighbours['tsSortLinIdx'] = np.argsort(np.array(Neighbours['lastTs']).T.flatten())[::-1]
        Neighbours['lastTs'][Neighbours['centerIdx']][Neighbours['centerIdx']] = \
            Neighbours['lastTs'][Neighbours['centerIdx']][Neighbours['centerIdx']] - 1

        # select associated neighbouring events
        Neighbours['associatedCount'] = 0
        Neighbours['isAssociated'] = np.zeros((OptFlowEstParams['L'], OptFlowEstParams['L']))
        Neighbours['isCollinear'] = np.zeros((OptFlowEstParams['L'], OptFlowEstParams['L']))
        # indicates the last selected neighbouring events that cause spatial collinearity of OptFlowEst['deltaP']
        Neighbours['inCoverage'] = np.zeros((OptFlowEstParams['L'], OptFlowEstParams['L']))
        # indicates the neighbouring events that are 8-neighbours of associated and collinear events

        # initialise Neighbours['inCoverage'] with 8-neighbours of the EOI
        NbPosRange['xl'] = max(Neighbours['centerIdx'] - 1, 0)
        NbPosRange['xh'] = min(Neighbours['centerIdx'] + 1, OptFlowEstParams['L'] - 1)
        NbPosRange['yl'] = max(Neighbours['centerIdx'] - 1, 0)
        NbPosRange['yh'] = min(Neighbours['centerIdx'] + 1, OptFlowEstParams['L'] - 1)

        isDirectNb = np.zeros((Neighbours['length'], Neighbours['length']))
        isDirectNb[NbPosRange['yl']: NbPosRange['yh'] + 1, NbPosRange['xl']: NbPosRange['xh'] + 1] = True
        isDirectNb[Neighbours['centerIdx'], Neighbours['centerIdx']] = False
        Neighbours['inCoverage'] = np.logical_or(Neighbours['inCoverage'], isDirectNb)

        # collinear check is required only if OptFlowEstParams['N_NB'] < OptFlowEstParams['L']
        CollinearCheck = {'isCollinear': (OptFlowEstParams['N_NB'] < OptFlowEstParams['L']), 'collinearTypeCount': 0,
                          'collinearTypeExists': np.zeros((1, 4))}

        OptFlowEst = {'deltaP': np.zeros((OptFlowEstParams['N_NB'], 2)),
                      'deltaTs': np.zeros((OptFlowEstParams['N_NB'], 1))}
        sortIdx = 2

        while Neighbours['associatedCount'] < OptFlowEstParams['N_NB'] and sortIdx <= Neighbours['count']:
            # linear index, which indicates the position, of the currently considered neighbouring event
            neighbourLinIdx = int(Neighbours['tsSortLinIdx'][sortIdx - 1])

            if (not Neighbours['isSamePolarity'].T.flatten()[neighbourLinIdx]) or \
                    (not Neighbours['inCoverage'].flatten()[neighbourLinIdx]) or \
                    (Neighbours['isAssociated'].flatten()[neighbourLinIdx]) or \
                    (Neighbours['isCollinear'].flatten()[neighbourLinIdx]):
                sortIdx = sortIdx + 1
                continue
            # next associated neighbour found, if not limited by spatial collinearity
            NewEvent['relativeP'] = Neighbours['relativeP'][neighbourLinIdx]
            NewEvent['deltaP'] = Neighbours['deltaP'][neighbourLinIdx]
            NewEvent['deltaTs'] = Neighbours['lastTs'].T.flatten()[neighbourLinIdx] - Eoi['ts']

            # update coverage
            NbPosRange['xl'] = int(max(NewEvent['relativeP'][0] - 1, 0))
            NbPosRange['xh'] = int(min(NewEvent['relativeP'][0] + 1, OptFlowEstParams['L'] - 1))
            NbPosRange['yl'] = int(max(NewEvent['relativeP'][1] - 1, 0))
            NbPosRange['yh'] = int(min(NewEvent['relativeP'][1] + 1, OptFlowEstParams['L'] - 1))

            isDirectNb = np.zeros((Neighbours['length'], Neighbours['length']))
            isDirectNb[NbPosRange['yl']: NbPosRange['yh'] + 1, NbPosRange['xl']: NbPosRange['xh'] + 1] = True
            isDirectNb[int(NewEvent['relativeP'][1])][int(NewEvent['relativeP'][0])] = False
            Neighbours['inCoverage'] = np.logical_or(Neighbours['inCoverage'].flatten(), isDirectNb.flatten())

            # collinearity check
            if CollinearCheck['isCollinear']:
                # check collinear arrangement type of newly selected event indicates a non-collinear arrangement
                # by default
                collinearType = 0

                if NewEvent['deltaP'][0] == 0:
                    collinearType = 1
                elif NewEvent['deltaP'][1] == 0:
                    collinearType = 2
                elif NewEvent['deltaP'][0] == NewEvent['deltaP'][1]:
                    collinearType = 3
                elif NewEvent['deltaP'][0] == -NewEvent['deltaP'][1]:
                    collinearType = 4

                # update collinearity check variables
                if collinearType == 0:
                    CollinearCheck['isCollinear'] = False
                elif not CollinearCheck['collinearTypeExists'][collinearType]:
                    CollinearCheck['collinearTypeExists'][collinearType] = True
                    if CollinearCheck['collinearTypeCount'] == 0:
                        CollinearCheck['collinearTypeCount'] = 1
                    else:
                        CollinearCheck['collinearTypeCount'] = 2
                        CollinearCheck['isCollinear'] = False
                # extend greedy selection if collinearity persists until
                # Neighbours['associatedCount'] == OptFlowEstParams['L'] - 1
                if CollinearCheck['isCollinear'] and Neighbours['associatedCount'] == OptFlowEstParams['N_NB'] - 1:
                    Neighbours['isCollinear'][neighbourLinIdx] = True
                    sortIdx = 2
                    continue
            Neighbours['associatedCount'] = Neighbours['associatedCount'] + 1
            Neighbours['isAssociated'][(neighbourLinIdx // len(Neighbours['isAssociated'][0]))][
                (neighbourLinIdx % len(Neighbours['isAssociated'][0]))] = True

            OptFlowEst['deltaTs'][Neighbours['associatedCount'] - 1] = NewEvent['deltaTs']
            OptFlowEst['deltaP'][Neighbours['associatedCount'] - 1] = NewEvent['deltaP']
            sortIdx = 2
        # filtering due to insufficient associated neighbouring events selected for non-iterative plane fitting
        if Neighbours['associatedCount'] < OptFlowEstParams['N_NB']:
            continue
        # estimate optical flow based on plane fitting/concept of directional derivative,
        # fits only the gradient of the plane. Only a 2x2 matrix inversion required.
        # Equation - delta ts = dot(grad ts, delta p)
        OptFlowEst['gradient'] = np.linalg.lstsq(OptFlowEst['deltaP'], OptFlowEst['deltaTs'])[0]
        OptFlowEst['gradientNorm'] = np.linalg.norm(OptFlowEst['gradient'])

        # goodness-of-fit noise rejection via plane support evaluation (valid events have the same polarity as the EOI)
        Neighbours['samePolDeltaTs'] = Neighbours['lastTs'].T[Neighbours['isSamePolarity'].T] - Eoi['ts']

        mul1 = np.array(Neighbours['deltaP'])[np.array(Neighbours['isSamePolarity'].T.flatten())]
        mul2 = OptFlowEst['gradient']

        OptFlowEst['samePolEstDeltaTs'] = np.matmul(mul1[:, [1, 0]], mul2.T[:, [1, 0]].T)
        OptFlowEst['samePolDeltaTsResidual'] = np.abs(
            Neighbours['samePolDeltaTs'] - OptFlowEst['samePolEstDeltaTs'].reshape(-1))
        OptFlowEst['planeSupportCount'] = np.sum(OptFlowEst['samePolDeltaTsResidual'] < OptFlowEstParams['E']) - 1

        # OptFlowEst['gradientNorm'] == 0 happens when OptFlowEst['deltaTs']  is a zero vector
        if OptFlowEst['gradientNorm'] == 0 or OptFlowEst['planeSupportCount'] < OptFlowEstParams['N_SP']:
            continue
        # this calculation prevents the issue of infinite velocity component
        OptFlowEst['normalisedGrad'] = OptFlowEst['gradient'] / OptFlowEst['gradientNorm']
        OptFlowEst['velocity'] = (1 / OptFlowEst['gradientNorm']) * OptFlowEst['normalisedGrad']

        # record critical information for optical flow evaluation
        OptFlowEval['velocities'][OptFlowEval['eventIdx']] = OptFlowEst['velocity'].flatten()
        OptFlowEval['x'][OptFlowEval['eventIdx']] = Eoi['x']
        OptFlowEval['y'][OptFlowEval['eventIdx']] = Eoi['y']
        OptFlowEval['p'][OptFlowEval['eventIdx']] = Eoi['p']
        OptFlowEval['ts'][OptFlowEval['eventIdx']] = Eoi['ts']
        OptFlowEval['eventIdx'] = OptFlowEval['eventIdx'] + 1

        if visualise_spatial_neighbourhood is True:
            plot_neighbourhood(Neighbours, OptFlowEst, Eoi)

    # trim optical flow vectors/matrices
    OptFlowEval['length'] = OptFlowEval['eventIdx'] - 1
    OptFlowEval['velocities'] = OptFlowEval['velocities'][0: OptFlowEval['length']]
    OptFlowEval['x'] = OptFlowEval['x'][0: OptFlowEval['length']]
    OptFlowEval['y'] = OptFlowEval['y'][0: OptFlowEval['length']]
    OptFlowEval['p'] = OptFlowEval['p'][0: OptFlowEval['length']]
    OptFlowEval['ts'] = OptFlowEval['ts'][0: OptFlowEval['length']]
    OptFlowEval['velocities'] = OptFlowEval['velocities'][:, [1, 0]]
    OptFlowEval['x'] = OptFlowEval['x'] + 1
    OptFlowEval['y'] = OptFlowEval['y'] + 1

    savemat("SOFEApy.mat", OptFlowEval)

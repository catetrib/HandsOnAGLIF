import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from load_eq import load_v3
import pickle
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from carica_exp_info import neuron_info_load
#from plotta_sim_dep_ini_Idip import plot_tr_v3_vect3_dep_ini,plot_tr_from_fit_neuron_rette_dep_ini

def plotta_t_spikes_e_num_spikes(corr_sp,t_sp_sim,spk_time,Istm):
    for i in range(corr_sp):
        plt.figure(200);    #PLOT degli spikes times per le diverse correnti
        plt.scatter(spk_time[i], Istm[len(Istm) - 1 - i] * np.ones(len(spk_time[i])), marker='|',
                        color='r',label='exp');
        plt.scatter(t_sp_sim[len(t_sp_sim) - 1 - i],
                        Istm[len(Istm) - 1 - i] * np.ones(len(t_sp_sim[len(t_sp_sim) - 1 - i])), marker='|',
                        color='b',label='sim');
        plt.ylabel('Current')
        plt.xlabel('spike times')

        plt.figure(201)   #PLOT NUMERO DI SPIKES IN FUNZIONE DELLA CORRENTE
        plt.scatter(Istm[len(Istm) - 1 - i], len(spk_time[i]), marker='*', color='r',label='exp')
        plt.scatter(Istm[len(Istm) - 1 - i], len(t_sp_sim[len(t_sp_sim) - 1 - i]), marker='*', color='b',label='sim')
        plt.ylabel('number of spikes')
        plt.xlabel('Current')

def calcola_spikes_al_variare_Iadap(Istm,alpha3,Idep_ini_vr,delta1,bet,psi1,time_scale,vrm,vth):
    t = sym.Symbol('t')
    delta, Psi, alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('delta,Psi,alpha,beta,gamma,IaA0,IdA0,t0,V0')
    [V, Iadap, Idep] = load_v3()
    IaA_min = 0
    m = 1000 #numero campioni

    time_sp = np.zeros([len(Istm), m])
    time_var = np.zeros([len(Istm), m - 1])
    for i in range(len(Istm)):
        IaA_max = Idep_ini_vr + alpha3[i] / bet + (delta1 / bet) * (1 + vrm)

        IaA = np.linspace(IaA_min, IaA_max, m)


        for j in range(len(IaA)):
            aux = V.subs(alpha, alpha3[i]).subs(beta, bet).subs(delta, delta1).subs(t0, 0).subs(V0, vrm).subs(IaA0,IaA[j]).subs(
            IdA0, Idep_ini_vr).subs(Psi, psi1)

            lam_x = sym.lambdify(t, aux, modules=['numpy'])
            #x_vals = np.linspace(0, 1000, 10001) / time_scale
            x_vals = np.linspace(0, 400, 401) / time_scale
            y_vals = lam_x(x_vals)
            aus = np.nonzero((y_vals > vth) * y_vals)
            if aus[0].size > 0:
                time_sp[i, j] = x_vals[aus[0][0]] * time_scale
            ##if j > 0:
            ##    time_var[i, j - 1] = (time_sp[i, j] - time_sp[i, j - 1]) / (IaA[j] - IaA[j - 1])
            else:
                time_sp[i, j] = -1

    return time_sp,IaA,IaA_min,IaA_max,m

def calcola_primi_spike_modello(Istm,alpha3,Idep_ini,delta1,bet,psi1,time_scale,Ith,vth):
    t = sym.Symbol('t')
    delta, Psi, alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('delta,Psi,alpha,beta,gamma,IaA0,IdA0,t0,V0')
    [V, Iadap, Idep] = load_v3()

    spt_00 = np.zeros(len(Istm))
    for i in range(len(Istm)):
        aux = V.subs(alpha, alpha3[i]).subs(beta, bet).subs(delta, delta1).subs(t0, 0).subs(V0, -1).subs(IaA0,0).subs(IdA0, Idep_ini * (Istm[i] - Ith) * (Istm[i] > Ith)).subs(Psi, psi1)
        lam_x = sym.lambdify(t, aux, modules=['numpy'])
        x_vals = np.linspace(0, 1000, 10001) / time_scale
        y_vals = lam_x(x_vals)
        aus = np.nonzero((y_vals > vth) * y_vals)  # aus contiene le posizioni in cui y_vals>vth
        if aus[0].size > 0:
            spt_00[i] = x_vals[
                            aus[0][0]] * time_scale  # salviamo in spt_00 il tempo del primo spike prodotto dal modello
        else:
            spt_00[i] = -1  # salviamo in spt_00 il valore -1 se la corrente analizzata non produce spike
    return spt_00

def calcola_monod(n_test,fun_loss_sel,bnds,bound):
    met = 'Nelder-Mead'  # 'slsqp'
    loss = []
    best_loss = np.inf
    print('mybound: ',np.array(bnds.ub[:].tolist()))
    for i in range(n_test):
        #aux = np.array(bnds.lb[:].tolist()) + np.random.rand(len(bnds.lb[:].tolist())) * (np.array(bnds.ub[:].tolist()) - np.array(np.array(bnds.ub[:].tolist())))
        aux = np.array(bnds.lb[:].tolist()) + np.random.rand(len(bnds.lb[:].tolist())) * (np.array(bnds.ub[:].tolist()) - np.array(np.array(bnds.lb[:].tolist())))
        if bound:
            res = minimize(fun_loss_sel, aux, method=met, bounds=bnds, options={'maxiter': 50000, 'disp': True})
        else:
            res = minimize(fun_loss_sel, aux, method=met, options={'maxiter': 50000, 'disp': True})
        loss.append(res.fun)
        if (res.fun < best_loss):
            best_loss = res.fun
            best_res = res
        print('res.fun: ',res.fun)
    return best_res,best_loss

def compute_block_functions(neuron_nb):

    def func_linear(x, a, b):
        return a * x + b

    def func_quad(x, a, b, c):
        return a * x * x + b * x + c

    def func_cubic(x, a, b, c, d):
        return a * x * x * x + b * x * x + c * x + d

    fitting_rule = 'monod'

    EL, vrm, vth, Istm, spk_time_orig, dur_sign, input_start_time, spk_time, spk_interval, is_spk_curr, is_spk_2_curr = neuron_info_load(
        neuron_nb)

    post_tempo_finale = np.zeros(len(Istm))
    is_blocked = np.array(np.zeros(Istm.__len__()), dtype=bool)

    for i in range(len(Istm)):

        if is_spk_curr[i]:
            if not is_spk_2_curr[i]:
                if (dur_sign - spk_time[i][0]) - (spk_time[i][0]) / 2 > 0:
                    post_tempo_finale[i] = 1.5 * spk_time[i][0]
                    is_blocked[i] = True

            else:
                if (dur_sign - spk_time[i][-1]) - (spk_time[i][-1] - spk_time[i][-2]) * 2 > 0:
                    is_blocked[i] = True
                    post_tempo_finale[i] = spk_time[i][-1] + (spk_time[i][-1] - spk_time[i][-2]) / 2

    import sympy as sym
    I = sym.Symbol('I')

    I_monod_inf = -np.inf
    I_monod_sup = np.inf

    # MODIFICATO DA EMILIANO

    Ind_first_current_spiking = np.where(is_spk_curr)[0][0]
    if is_blocked[Ind_first_current_spiking]:  # se la corrente meno intensa, che spara, si blocca

        if np.where(is_blocked == False)[0].size > 0:  # se ci sono correnti che non si bloccano
            Ind_last_current_inf_blocked = np.where(is_blocked == False)[0][0] - 1;
            I_monod_inf = (Istm[Ind_last_current_inf_blocked] + Istm[Ind_last_current_inf_blocked + 1]) / 2
            t_val_min = I * 0 + post_tempo_finale[Ind_first_current_spiking]
        else:
            Ind_last_current_inf_blocked = len(Istm) / 2;
            Ind_first_current_sup_blocked = (len(Istm) / 2) + 1;

        if Ind_first_current_spiking < Ind_last_current_inf_blocked :  # se oltre alla meno intensa c'è almeno un'altra corrente per cui lo spiking si blocca
            [par, op] = curve_fit(func_linear,
                                  Istm[range(Ind_first_current_spiking, Ind_last_current_inf_blocked + 1)].tolist(),
                                  post_tempo_finale[range(Ind_first_current_spiking,
                                                          Ind_last_current_inf_blocked + 1)].tolist())  # evitare che fitti gli 0 finali
            t_val_min = I * par[0] + par[1]


    else:
        t_val_min = np.inf + I
        I_monod_inf = -np.inf

    if is_blocked[-1] > 0:  # se per la corrente più intensa lo spiking si blocca

        if np.where(is_blocked == False)[0].size > 0:  # se ci sono correnti per cui lo spiking non si blocca
            Ind_first_current_sup_blocked = np.where(is_blocked == False)[0][- 1] + 1

            t_val_max = I * 0 + post_tempo_finale[Ind_first_current_sup_blocked]

        if Ind_first_current_sup_blocked < len(
                Istm) - 1:  # se oltre alla più intensa c'è almeno un'altra corrente per cui lo spiking si blocca
            [par, op] = curve_fit(func_linear, Istm[range(Ind_first_current_sup_blocked, Istm.size)].tolist(),
                                  post_tempo_finale[range(Ind_first_current_sup_blocked, Istm.size)].tolist())
            t_val_max = I * par[0] + par[1]
        I_monod_sup = (Istm[Ind_first_current_sup_blocked] + Istm[Ind_first_current_sup_blocked - 1]) / 2
    else:
        t_val_max = np.inf + I

    block_file = open(neuron_nb + "_block_functions.txt", mode="w", encoding="utf-8")
    block_file.write("I_monod_sup = " + str(I_monod_sup) + "\n")
    block_file.write("I_monod_inf = " + str(I_monod_inf) + "\n")
    block_file.write("t_val_max   = " + str(t_val_max) + "\n")
    block_file.write("t_val_min   = " + str(t_val_min) + "\n")
    block_file.close()

    with open('block_info_' + neuron_nb + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([I_monod_inf,I_monod_sup,t_val_min,t_val_max], f)



    return(I_monod_inf,I_monod_sup,t_val_min,t_val_max)

def print_figure(neuron_nb,monod_active):
    def monod(x, a, b, c, alp):
        return c + (a * np.exp(b) * x) / (alp + x)
    k=1000

    with open('neuron_' + neuron_nb + '_model_parameter.pkl','rb') as f:  # Python 3: open(..., 'wb')
        [bet, delta1, sc, time_scale, Idep_ini_vr, tao_m, Ith, Idep_ini,Cm] = pickle.load(f)
    EL, vrm, vth, Istm, spk_time_orig, dur_sign, input_start_time, spk_time, spk_interval, is_spk_curr, is_spk_2_curr = neuron_info_load(
        neuron_nb)
    Vconvfact=-EL
    alpha=Istm/sc
    t_sp_sim = np.empty((Istm.__len__()),object)
    if monod_active:
        with open('best_res_' + neuron_nb + '.pkl', 'rb') as f:
            [best_res] = pickle.load(f)
        with open('block_info_' + neuron_nb + '.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
            [I_monod_inf, I_monod_sup, t_val_min, t_val_max]=pickle.load( f)

    xdata = np.linspace(0, int(input_start_time + dur_sign) + 30, 100)
    plt.figure();
    popt = np.array([0., 0., 0., 0.])
    with open('Iadap_time_selected_' + neuron_nb + '.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
        [t_sp_abs_tutti,Iada_tutti]=pickle.load(f)
    for i in range(len(Istm)):
        t_sp_sim[i]=[]
        if is_spk_curr[i]:
            print("first Trace")

            if not monod_active:
                if not is_spk_2_curr[i]:
                    [t_aux, time, voltage,Iada_final] = plot_tr_v3_vect3_dep_ini(Vconvfact, vth, vrm, [], bet, delta1,alpha[i], sc, time_scale, Idep_ini_vr,input_start_time, dur_sign,Idep_ini,Ith)
                else:
                    [t_aux, time, voltage,Iada_final] = plot_tr_v3_vect3_dep_ini(Vconvfact, vth, vrm, Iada_tutti[i], bet, delta1,alpha[i], sc, time_scale, Idep_ini_vr,input_start_time, dur_sign,Idep_ini,Ith)
                # cate:
                t_sp_sim[i]=t_aux
                if is_spk_2_curr[i]:
                    plt.scatter(t_sp_abs_tutti[i], Iada_tutti[i])
            else:

                popt[0] = best_res.x[0]
                popt[1] = best_res.x[1] * Istm[i] / 1000
                popt[2] = best_res.x[2]
                popt[3] = best_res.x[3]

                aux = Istm / sc
                alpha = aux.tolist()

                fun_sel = monod
                [t_aux, ada_aux, time, voltage] = plot_tr_from_fit_neuron_rette_dep_ini(Vconvfact, vth, vrm, fun_sel,
                                                                                        popt, bet, delta1, alpha[i],
                                                                                        sc, time_scale, Idep_ini_vr,
                                                                                        input_start_time, dur_sign,
                                                                                        t_val_min, I_monod_inf,
                                                                                        t_val_max, I_monod_sup,
                                                                                        Idep_ini, Ith)  # np.inf

                t_sp_sim[i]=t_aux
                if is_spk_2_curr[i]:

                    plt.plot(xdata, fun_sel(xdata, *popt),
                         label='fit: a=%5.3f, b=%5.3f,c=%5.3f,alpha=%5.3f' % tuple(popt))
                    plt.scatter(t_sp_abs_tutti[i], Iada_tutti[i])
                    plt.legend( loc='lower right')
                   
    plt.title('neuron ' + neuron_nb)
    plt.xlabel('Time (ms)')
    plt.ylabel('Iadap')
    plt.xlim(0, dur_sign)

                # plt.title('Iada_tot Idep' + str(Idep_ini_vr) + ' tao ' + str(tao_m) + ' Ith ' + str(Ith))
    if not monod_active:
        plt.savefig('punti_' + neuron_nb + '.png')
    else:
        plt.savefig('MONOD_punti_' + neuron_nb + '.png')

    plt.close()

    for i in range(len(Istm)):
        print(Istm[i])
        lo=plt.scatter(spk_time[i], Istm[i] * np.ones(len(spk_time[i])), marker='|',
                    label='exp',color='r',alpha=0.5);
        ll=plt.scatter(t_sp_sim[i],Istm[i] * np.ones(len(t_sp_sim[i])), marker='|',label='sim',color='b',alpha=0.5);
    plt.xlim(0, dur_sign)

    #plt.ylim(min(Istm)-(max(Istm)-min(Istm))/Istm.__len__(), max(Istm)+(max(Istm)-min(Istm))/Istm.__len__())
    plt.legend((lo, ll),
                       ('exp', 'sim'),
                       scatterpoints=1,
                       loc='upper center', bbox_to_anchor=(0.1, -0.05),
                       ncol=3,
                       fontsize=8)
    plt.title('neuron ' + neuron_nb)
    plt.ylabel('Current (pA)')
    plt.xlabel('spike times (ms)')



    if not monod_active:
        plt.savefig('raster_plot_' + neuron_nb + '.png')
            # plt.savefig('sper_raster_plot_' + neuron_nb + '.png')
    else:
        plt.savefig('MONOD_raster_plot_' + neuron_nb + '.png')
    plt.close()
    plt.figure(k + 20)
    for i in range(len(Istm)):

        if is_spk_curr[i]:

            lo=plt.scatter(Istm[i], len(spk_time[i]), marker='*', color='r',alpha=0.5)
            ll=plt.scatter(Istm[i], len(t_sp_sim[i]), marker='*', color='b',alpha=0.5)
            plt.legend((lo, ll),
                       ('exp', 'sim'),
                       scatterpoints=1,
                       loc='upper center', bbox_to_anchor=(0.1, -0.05),
                       ncol=3,
                       fontsize=8)
    plt.title('neuron ' + neuron_nb)
    plt.ylabel('number of spikes')
    plt.xlabel('Current (pA)')
    if not monod_active:
        plt.savefig('spike_numbers_' + neuron_nb + '.png')

    else:
        plt.savefig('MONOD_spike_numbers_' + neuron_nb + '.png')

    plt.close()
def plot_tr_v3_vect3_dep_ini(Vconvfact,vtm,vrm,a_inc,bet,delta1,alpha3,sc, tao_m, Idep_ini_vr, st_sign, dur_sign,Idep_ini,Ith):

    #import pyabf
    import matplotlib.pyplot as plt
    import sympy as sym
    import numpy as np
    from load_eq import load_v3

    plotta=False
    n_sp=np.size(a_inc)

    corr = round(alpha3 * sc) / 1000

    tim_aux=np.linspace(0, 1000, 10000)

    #alpha3 = [0.252587 * 4, 0.252587 * 5]
    t_out = []
    tr=int(alpha3/sc)+10
    #cell_num = '95810005'

    t = sym.Symbol('t')
    delta, Psi, alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('delta,Psi,alpha,beta,gamma,IaA0,IdA0,t0,V0')

    [V, Iadap, Idep] = load_v3()
    psi1 = ((-4) * bet + ((1 + delta1) ** 2)) ** (0.5)

    t0_val=0
    #vtm=np.mean(vol_tr)/66.35
    #vrm=-1.0
    #d_in=st_point_dep1[r]
    #a_in=st_point_ada1[r]
    #aux = V.subs(alpha, alpha3).subs(beta, bet).subs(gamma, gam).subs(t0, t0_val).subs(V0, -1).subs(IaA0, a_in).subs(IdA0,d_in)
    #aux = V.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t0_val).subs(V0, -1).subs(IaA0,a_in).subs(IdA0,d_in).subs(Psi, psi1)
    if plotta:
        plt.figure()
    time=[]
    voltage=[]
    t_next=0
    adap_final=[]
    for i in range(n_sp+1):
        if i>0:
            t_init=t_next+2/tao_m
        else:
            t_init=t_next


        #aux = V.subs(alpha, alpha3).subs(beta, bet).subs(gamma, gam).subs(t0,t_init ).subs(V0, vrm).subs(IaA0, ada+a_inc).subs(IdA0,dep+d_inc)
        if i==0:
            aux = V.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, -1).subs(IaA0,0).subs(IdA0, Idep_ini*(corr*1000-Ith)*(corr*1000>Ith)).subs(Psi, psi1)
        else:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('***** ada inc ****' + str(i) + ' *****spike****')
            print(a_inc[i-1])
            aux = V.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, vrm).subs(IaA0, a_inc[i-1]).subs(IdA0, Idep_ini_vr).subs(Psi, psi1)

        #print('ada comp')
        #print( ada+a_inc)
        #print('dep comp')
        #print(dep + d_inc)

#        print('ada')
#        print(ada_nc)
#        print('dep')
#        print(dep_nc)
        lam_x = sym.lambdify(t, aux, modules=['numpy'])
        x_vals = tim_aux[(tim_aux / tao_m)>t_init]/ tao_m
        y_vals = lam_x(x_vals)
    # plt.plot(31.1 + x_vals * 15.58, y_vals * 66.35)


        if len(np.nonzero(y_vals > vtm)[0]) > 0:



            t_next=x_vals[y_vals>vtm][0]
            t_out.append(t_next*tao_m)

            if i > 0:
                adap_final.append(Iadap.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, vrm).subs(IaA0,a_inc[i - 1]).subs(IdA0, Idep_ini_vr).subs(Psi, psi1).subs(t, t_next))
            else:
                adap_final.append(Iadap.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, -1).subs(IaA0, 0).subs(IdA0, Idep_ini*(corr*1000-Ith)*(corr*1000>Ith)).subs(Psi, psi1).subs(t, t_next))


#            ada = Iadap.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, vrm).subs(IaA0,ada+a_inc[i]).subs(IdA0, dep+d_inc[i]).subs(Psi, psi1).subs(t, t_next)
#            dep = Idep.subs(beta, bet).subs(IdA0, dep+d_inc[i]).subs(t, t_next).subs(t0, t_init)
            #ada = Iadap.subs(alpha, alpha3).subs(beta, bet).subs(gamma, gam).subs(t0, t_init).subs(V0, vrm).subs(IaA0,ada+a_inc).subs(IdA0, dep+d_inc).subs(t, t_next)
            #dep = Idep.subs(gamma, gam).subs(t0, t_init).subs(IdA0, dep+d_inc).subs(t, t_next)
#        auxa = (IaA - a_in) ** 2
#        auxd = (IdA - d_in) ** 2
#        ada_nc = Iadap_mat[4, auxa.argmin(), auxd.argmin()]
#        dep_nc = Idep_mat[4, auxa.argmin(), auxd.argmin()]
            x_sector = np.logical_and(t_init < x_vals, x_vals < t_next) * x_vals


            x_vals = x_sector[np.nonzero(x_sector)]
            y_vals = lam_x(x_vals)
            if plotta:
                plt.plot(st_sign + x_vals * tao_m, y_vals * Vconvfact)
            time.append(st_sign + x_vals * tao_m)
            voltage.append(y_vals * Vconvfact)
        else:
            print('aoa')
            print(n_sp)
            print(len(np.nonzero(y_vals > vtm)[0]))
            print(vtm)
            t_next = dur_sign/tao_m
            x_sector = np.logical_and(t_init < x_vals, x_vals < t_next) * x_vals
            x_vals = x_sector[np.nonzero(x_sector)]
            y_vals = lam_x(x_vals)
            if plotta:
                plt.plot(st_sign + x_vals * tao_m, y_vals * Vconvfact)
            time.append(st_sign + x_vals * tao_m)
            voltage.append(y_vals * Vconvfact)

            i=n_sp+1
        t_init = t_next + 2 / tao_m
    #plt.plot(tim, vol)
    #plt.title('trace ('+ str(sc*alpha3)+'nA)')
    return t_out,time,voltage,adap_final


def plot_tr_from_fit_neuron_rette_dep_ini(Vconvfact, vtm, vrm, funzione,popt, bet, delta1,alpha3, sc,tao,Idep_ini_vr,st_sign,dur_sign,lin_func_inf,vinc_inf,lin_func_sup,vinc_sup,Idep_ini,Ith):

    #import pyabf
    import matplotlib.pyplot as plt
    import sympy as sym
    import numpy as np
    from load_eq import load_v3
    import sympy as sym
    I = sym.Symbol('I')
    plotta=False
    #n_sp=np.size(a_inc)

    corr=round(alpha3*sc)/1000
    if corr*1000<=vinc_inf:
        dur_sign=min(dur_sign,lin_func_inf.subs(I,corr*1000))

    if corr*1000>=vinc_sup:
        dur_sign=min(dur_sign,lin_func_sup.subs(I,corr*1000))
    tim = []


    tim_aux = np.linspace(0, 1000, 10000)
    #alpha3 = [0.252587 * 4, 0.252587 * 5]
    t_out = []
    #cell_num = '95810005'

    t = sym.Symbol('t')
    delta, Psi, alpha, beta, gamma, IaA0, IdA0, t0, V0 = sym.symbols('delta,Psi,alpha,beta,gamma,IaA0,IdA0,t0,V0')

    [V, Iadap, Idep] = load_v3()
    psi1 = ((-4) * bet + ((1 + delta1) ** 2)) ** (0.5)

    t_init=0
    if plotta:
        plt.figure()
    i=0
    ada_vec=[]
    time=[]
    voltage=[]
    #while i<20:
    while t_init * tao < dur_sign :
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('***** ada  ****' + str(i) + ' *****spike****')
        if i==0:
            print(funzione(t_init * tao, *popt))
        else:
            new_ada=funzione(t_next*tao,*popt)
            ada_vec.append(new_ada)
            print(new_ada)
            print(funzione(t_next*tao,*popt))

        print('t_init')
        print(t_init)
        print(t_init * tao)


        # aux = V.subs(alpha, alpha3).subs(beta, bet).subs(gamma, gam).subs(t0,t_init ).subs(V0, vrm).subs(IaA0, ada+a_inc).subs(IdA0,dep+d_inc)
        if t_init * tao== dur_sign:
            aux = V.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, vrm).subs(IaA0,adaf).subs(IdA0, depf).subs(Psi, psi1)

        else:
            if i==0:
                aux = V.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, -1).subs(IaA0,0).subs(IdA0,Idep_ini*(corr*1000-Ith)*(corr*1000>Ith)).subs(Psi, psi1)

            else:
                aux = V.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, vrm).subs(IaA0, new_ada).subs(IdA0, Idep_ini_vr).subs(Psi, psi1)


        lam_x = sym.lambdify(t, aux, modules=['numpy'])
        x_vals = tim_aux[(tim_aux / tao) > t_init] / tao
        y_vals = lam_x(x_vals)
        # plt.plot(31.1 + x_vals * 15.58, y_vals * 66.35)


        if len(np.nonzero(y_vals > vtm)[0]) > 0:

            t_next = x_vals[y_vals > vtm][0]#+t_init
            if i == 0:
                adaf = Iadap.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, vrm).subs(IaA0, 0).subs(IdA0, Idep_ini*(corr*1000-Ith)*(corr*1000>Ith)).subs(Psi, psi1).subs(t, t_next)
                depf = Idep.subs(beta, bet).subs(IdA0, Idep_ini*(corr*1000-Ith)*(corr*1000>Ith)).subs(t0, t_init).subs(t, t_next)
            else:
                adaf = Iadap.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0,vrm).subs(IaA0, new_ada).subs(IdA0, Idep_ini_vr).subs(Psi, psi1).subs(t, t_next)
                depf = Idep.subs(beta, bet).subs(IdA0, Idep_ini_vr).subs(t0, t_init).subs(t, t_next)
            print('.................adap....')
            print(adaf)
            print('..................dep....')
            print(depf)
            if (t_next*tao<dur_sign):
                t_out.append(t_next*tao)
                print("t_next .....")
                print(t_next)
                print(t_next*tao)
            else:
                t_next=dur_sign/tao

            x_sector = np.logical_and(t_init < x_vals, x_vals < t_next) * x_vals

            x_vals = x_sector[np.nonzero(x_sector)]
            y_vals = lam_x(x_vals)
            time.append(st_sign + x_vals * tao)
            voltage.append(y_vals * Vconvfact)
            if plotta:
                plt.plot(st_sign + x_vals * tao, y_vals * Vconvfact)
            #st_sign=st_sign#+t_next*15.58
            print("st_sign .....")
            print(st_sign)
        else:
            print('aoa')
            print(i)
            print(len(np.nonzero(y_vals > vtm)[0]))
            print(vtm)
            t_next = dur_sign/tao
            x_sector = np.logical_and(t_init < x_vals, x_vals < t_next) * x_vals
            x_vals = x_sector[np.nonzero(x_sector)]
            y_vals = lam_x(x_vals)
            try:
                time.append(st_sign + x_vals * tao)
                voltage.append(y_vals * Vconvfact)
                if plotta:
                    plt.plot(st_sign + x_vals * tao, y_vals * Vconvfact)
            except:
                time.append(st_sign + x_vals * tao)
                voltage.append(y_vals * Vconvfact*np.ones(x_vals.size))
                if plotta:
                    plt.plot(st_sign + x_vals * tao, y_vals * Vconvfact*np.ones(x_vals.size))

        i = i + 1
        print('t_next')
        print(t_next)
        if t_next==dur_sign:
            adaf=Iadap.subs(alpha, alpha3).subs(beta, bet).subs(delta, delta1).subs(t0, t_init).subs(V0, vrm).subs(IaA0,new_ada ).subs(IdA0, Idep_ini_vr).subs(Psi, psi1).subs(t,dur_sign)
            depf=Idep.subs(beta, bet).subs(IdA0, Idep_ini_vr).subs(t0, t_init).subs(t,dur_sign)
            t_init=t_next
            vrm=y_vals[len(y_vals)-1]
            alpha3=0

        else:
            t_init = t_next + 2 / tao
    try:
        if plotta:
            plt.plot(tim, vol)
    except:
        print('no trace')
    if plotta:
        plt.title('trace (' + str(alpha3*sc/1000) + 'nA)')
    return t_out,ada_vec,time,voltage




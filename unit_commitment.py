# -*- coding: utf-8 -*-
# @Time    : 2019/4/19 9:19
# @Author  : ruyueshuo
# @File    : unit_commitment.py
# @Software: PyCharm
"""
This example formulates and solves the following simple UC model:
 minimize
       objective: min f(x)
 subject to
       constraints: AX<=b
 with
       variables: x0,x1,...,xn

This is a deterministic model solver and just a baseline.
If you want a stochastic programme model solver, try scenario generation method.

In this model, the variables are:
on-off status of thermal unit, power output of thermal unit, hydro unit, wind farm, solar station and battery station.
The following constraints are included:
Power balance constraints, upper and lower outputs constraints, ramp constraints, reservoir volume constraints,
on-off minimum time constraints, and reserve capacity constraints.
The objectives are:
Operation cost of thermal units and punishment for abandoning renewable energy.

There are three simplifications:
First one is the hydro power output has no relationship with reservoir volume or water-head height,
and the upcoming and up-flowing water volumes are the same or linear relationship during the whole 24h.
You can also set the reservoir volumes by yourself.
Second one is that the range of battery station power output is defined as [-Ps,Pu].
But if you consider pumped storage power station rather than battery station, the output range should be {-Ps,[0,Pu]}.
Third one is the start up cost of thermal unit is not concluded in the objective.

Version:
python -- 3.6
numpy -- 1.14.5
pandas -- 0.24.1
matplotlib -- 3.0.3
"""
# ============================== Import Libs =================================
import numpy as np
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt


class UnitCommitment(object):

    def __init__(self,
                 thermal_gen_info,
                 hydro_gen_info,
                 demand,
                 wind_predict_power,
                 solar_predict_power,
                 SCHEDULE_LEN,  # 24 for day ahead power schedule.
                 SEASON_FACTOR,  # [0, 1]. Larger value means flooded season and smaller value  means drawdown season.
                 RESERVE_FACTOR,  # Certain percentage of demand power.
                 ENS,  # Punishment factor for abandoning wind or solar.
                 NUM_THERMAL_GEN,
                 NUM_HYDRO_GEN):

        self.thermal_gen_info = thermal_gen_info
        self.hydro_gen_info = hydro_gen_info
        self.demand = demand
        self.wind_predict_power = wind_predict_power
        self.solar_predict_power = solar_predict_power
        self.SCHEDULE_LEN = SCHEDULE_LEN
        self.SEASON_FACTOR = SEASON_FACTOR
        self.RESERVE_FACTOR = RESERVE_FACTOR
        self.ENS = ENS
        self.NUM_THERMAL_GEN = NUM_THERMAL_GEN
        self.NUM_HYDRO_GEN = NUM_HYDRO_GEN
        self.m = Model('UnitCommitment')  # Create a model

    def model(self):
        """Create a new model then optimize it."""
        # ============================== Variables ================================
        # Create variables
        onoff_ind = {}  # thermal_onoff_n_t
        for i in range(self.NUM_THERMAL_GEN):
            for j in range(self.SCHEDULE_LEN):
                onoff_ind[i, j] = (self.m.addVar(vtype=GRB.BINARY, name='onoff_ind_' + str(i) + str(j)))

        thermal_power = {}  # thermal_power_n_t
        for i in range(self.NUM_THERMAL_GEN):
            for j in range(self.SCHEDULE_LEN):
                thermal_power[i, j] = self.m.addVar(vtype=GRB.CONTINUOUS, name='thermal_power_' + str(i) + '_' + str(j))

        hydro_power = {}  # hydro_power_n_t
        for i in range(self.NUM_HYDRO_GEN):
            for j in range(self.SCHEDULE_LEN):
                hydro_power[i, j] = self.m.addVar(lb=self.hydro_gen_info.Pmin[i], ub=self.hydro_gen_info.Pmax[i],
                                                  vtype=GRB.CONTINUOUS, name='hydro_power_' + str(i) + '_' + str(j))
        wind_power = []
        for j in range(self.SCHEDULE_LEN):
            wind_power.append(self.m.addVar(lb=0, ub=self.wind_predict_power[j], vtype=GRB.CONTINUOUS,
                                            name='wind_power_' + str(j)))
        wind_abandoned_power = []
        for j in range(self.SCHEDULE_LEN):
            wind_abandoned_power.append(self.m.addVar(lb=0, ub=self.wind_predict_power[j], vtype=GRB.CONTINUOUS,
                                                      name='wind_abandoned_power_' + str(j)))
        solar_power = []
        for j in range(self.SCHEDULE_LEN):
            solar_power.append(self.m.addVar(lb=0, ub=self.solar_predict_power[j], vtype=GRB.CONTINUOUS,
                                             name='solar_power_' + str(j)))
        battery_power = []
        # battery_status = []
        # for i in range(NUM_BATTERY):
        # for j in range(SCHEDULE_LEN):
        #     battery_status.append(m.addVar(vtype=GRB.BINARY, name='battery_status_' + str(j)))
        # for i in range(NUM_BATTERY):
        for j in range(self.SCHEDULE_LEN):
            battery_power.append(self.m.addVar(lb=-150, ub=200, vtype=GRB.CONTINUOUS, name='battery_power_' + str(j)))

        # ============================== Constraints ================================
        # Add constraint: power balance
        for i in range(self.SCHEDULE_LEN):
            thermal_gen_sum = quicksum(onoff_ind[j, i] * thermal_power[j, i] for j in range(self.NUM_THERMAL_GEN))
            hydro_gen_sum = quicksum(hydro_power[j, i] for j in range(self.NUM_HYDRO_GEN))
            self.m.addConstr(thermal_gen_sum + hydro_gen_sum + wind_power[i] + solar_power[i] + battery_power[i]
                             == self.demand[i], 'e' + str(i))
            self.m.addConstr(wind_abandoned_power[i] + wind_power[i] == self.wind_predict_power[i])

        # Add constraint: upper and lower outputs constraints
        for i in range(self.NUM_THERMAL_GEN):
            for j in range(self.SCHEDULE_LEN):
                self.m.addConstr(thermal_power[i, j] <= onoff_ind[i, j] * self.thermal_gen_info.Pmax[i])
                self.m.addConstr(thermal_power[i, j] >= onoff_ind[i, j] * self.thermal_gen_info.Pmin[i])

        # Add constraint: ramp constraints
        # thermal units
        for i in range(self.NUM_THERMAL_GEN):
            for j in range(self.SCHEDULE_LEN - 1):
                self.m.addConstr(
                    thermal_power[i, j + 1] - thermal_power[i, j] <= ((onoff_ind[i, j + 1] - onoff_ind[i, j]) *
                                                                      self.thermal_gen_info.Pmin[i] +
                                                                      self.thermal_gen_info.r[i] *
                                                                      self.thermal_gen_info.Pmax[i] * 60))
                self.m.addConstr(
                    thermal_power[i, j] - thermal_power[i, j + 1] <= ((onoff_ind[i, j] - onoff_ind[i, j] + 1) *
                                                                      self.thermal_gen_info.Pmin[i] +
                                                                      self.thermal_gen_info.r[i] *
                                                                      self.thermal_gen_info.Pmax[i] * 60))
        # hydro units
        for i in range(self.NUM_HYDRO_GEN):
            for j in range(self.SCHEDULE_LEN - 1):
                self.m.addConstr(hydro_power[i, j + 1] - hydro_power[i, j] <=
                                 self.hydro_gen_info.r[i] * self.hydro_gen_info.Pmax[i] * 60)
                self.m.addConstr(hydro_power[i, j + 1] - hydro_power[i, j] >=
                                 -(self.hydro_gen_info.r[i] * self.hydro_gen_info.Pmax[i] * 60))

        # Add constraint: hydro power station volume constraints
        volume = {}
        for i in range(self.NUM_HYDRO_GEN):
            volume[i, 0] = (self.hydro_gen_info.Vmin[i] + self.hydro_gen_info.Vmax[i]) / 2  # initial station volume
            for j in range(1, self.SCHEDULE_LEN):
                volume[i, j] = volume[i, j - 1] + self.hydro_gen_info.Pmax[i] * self.SEASON_FACTOR / 10 - hydro_power[
                    i, j - 1] / 10
                self.m.addConstr(volume[i, j] >= self.hydro_gen_info.Vmin[i])
                self.m.addConstr(volume[i, j] <= self.hydro_gen_info.Vmax[i])

        # Add constraint: pumped power station reservoir volume constraints
        volume_p = {0: (200 + 50) / 2}  # initial reservoir volume, suppose max vol:200, min vol:50.
        for j in range(1, self.SCHEDULE_LEN):
            # volume_p[j] = volume_p[j - 1] - battery_power[j - 1] / 10  # need to be changed
            volume_p[j] = volume_p[j - 1] - battery_power[j - 1] / 10
            self.m.addConstr(volume_p[j] >= 50)
            self.m.addConstr(volume_p[j] <= 200)

        # Add constraint: on-off minimum time
        '''The current method dosen't consider the status before schedule time t0.'''
        for i in range(self.NUM_THERMAL_GEN):
            for j in range(self.SCHEDULE_LEN - 1):
                # indicator will be 1 only when switched on
                indicator = onoff_ind[i, j + 1] - onoff_ind[i, j]
                ran = range(j + 1, min(self.SCHEDULE_LEN, j + self.thermal_gen_info.Tonmin[i]))
                # Constraints will be redundant unless indicator = 1
                for r in ran:
                    self.m.addConstr(onoff_ind[i, r] >= indicator)

        for i in range(self.NUM_THERMAL_GEN):
            for j in range(self.SCHEDULE_LEN - 1):
                # indicator will be 1 only when switched off
                indicator = onoff_ind[i, j] - onoff_ind[i, j + 1]
                ran = range(j + 1, min(self.SCHEDULE_LEN, j + self.thermal_gen_info.Tonmin[i]))
                # Constraints will be redundant unless indicator = 1
                for r in ran:
                    self.m.addConstr(onoff_ind[i, r] <= 1 - indicator)

        # Add constraint: reserve capacity
        for j in range(self.SCHEDULE_LEN):
            self.m.addConstr(
                quicksum((thermal_power[i, j] - onoff_ind[i, j] * self.thermal_gen_info.Pmin[i]) for i in
                         range(self.NUM_THERMAL_GEN)) +
                quicksum((hydro_power[h, j] - self.hydro_gen_info.Pmin[h]) for h in range(self.NUM_HYDRO_GEN)) >=
                self.demand[j] * self.RESERVE_FACTOR)
            self.m.addConstr(
                quicksum((onoff_ind[i, j] * self.thermal_gen_info.Pmax[i] - thermal_power[i, j]) for i in
                         range(self.NUM_THERMAL_GEN)) +
                quicksum((self.hydro_gen_info.Pmax[h] - hydro_power[h, j]) for h in range(self.NUM_HYDRO_GEN)) >=
                self.demand[j] * self.RESERVE_FACTOR)

        """
        ============================ Objective =============================
        Set objective : operation cost calculation & punishment for abandoning renewable energy
        Start up cost is not considered, here are some clue.
        start_state = last_start_state
        on_idx = np.where((on_index[0: num_gen] > 0) & (start_state[0:num_gen] < 0))
        on_idx = np.array(on_idx)[0]
        cost_start = (quicksum(gen_info.e[on_idx] * np.exp(gen_info.g[on_idx] * start_state[on_idx])) +
                      quicksum(gen_info.f[on_idx] * np.exp(gen_info.h[on_idx] * start_state[on_idx])))
        """
        self.m.setObjective(
            quicksum(
                self.thermal_gen_info.a[j] * (thermal_power[j, i] * thermal_power[j, i]) + self.thermal_gen_info.b[j] *
                thermal_power[j, i] + self.thermal_gen_info.c[j] for i in range(self.SCHEDULE_LEN) for j in
                range(self.NUM_THERMAL_GEN)) +
            self.ENS * quicksum((wind_predict_power[i] - wind_power[i] + solar_predict_power[i] - solar_power[i])
                                for i in range(self.SCHEDULE_LEN)), GRB.MINIMIZE)

        self.m.optimize()

    # @staticmethod
    def print_results(self):
        """Print results"""
        for v in self.m.getVars():
            print('%s %g' % (v.varName, v.x))

        print('Obj: %g' % self.m.objVal)
        # print(self.m.getAttr(GRB.Attr.X, self.m.getVars()))

    def plot_figs(self):
        """Plot figures"""
        x = np.arange(0, 24, 1)
        labels = ["Thermal", "Hydro", "Solar", "Wind", "Generation", "Storage"]
        thermal = np.zeros(self.SCHEDULE_LEN)
        hydro = np.zeros(self.SCHEDULE_LEN)
        solution = self.m.getAttr('X', self.m.getVars())

        for j in range(self.SCHEDULE_LEN):
            temp_t = 0
            for i in range(self.NUM_THERMAL_GEN, 2 * self.NUM_THERMAL_GEN, 1):
                temp_t += solution[i * self.SCHEDULE_LEN + j] * solution[
                    (i - self.NUM_THERMAL_GEN) * self.SCHEDULE_LEN + j]
            thermal[j] = temp_t

            temp_h = 0
            for h in range(2 * self.NUM_THERMAL_GEN, 2 * self.NUM_THERMAL_GEN + self.NUM_HYDRO_GEN, 1):
                temp_h += solution[h * self.SCHEDULE_LEN + j]
            hydro[j] = temp_h

        wind = self.wind_predict_power
        solar = self.solar_predict_power
        battery = [solution[b] for b in range(-self.SCHEDULE_LEN, 0, 1)]
        battery_g = [max(battery[t], 0) for t in range(self.SCHEDULE_LEN)]
        battery_s = [min(battery[t], 0) for t in range(self.SCHEDULE_LEN)]

        plt.stackplot(x, thermal, hydro, solar, wind, battery_g, battery_s, labels=labels)
        plt.plot(self.demand, label='Demand')
        plt.legend(loc="upper left")
        plt.xlabel('Hour')
        plt.ylabel('Power')
        plt.xlim(0, 23)
        plt.show()


if __name__ == '__main__':
    # e,g.
    # ============================== Import Data =================================
    # Input all the data and parameters here
    thermal_gen_info = pd.read_csv("gen_info/thermal_gen_info_HN.csv", sep=',')
    hydro_gen_info = pd.read_csv("gen_info/hydro_gen_info_HN.csv", sep=',')
    demand = np.array(pd.read_csv("demand_info/demand_info_2.csv")['demand'], dtype=float)
    wind_predict_power = np.array(pd.read_csv("gen_info/wind_predict_power.csv", sep=',')['wind']).astype(float)
    solar_predict_power = np.array(pd.read_csv("gen_info/wind_predict_power.csv", sep=',')['solar2']).astype(float)

    # If you wanna a single solution, run the following code.
    m = UnitCommitment(thermal_gen_info=thermal_gen_info,
                       hydro_gen_info=hydro_gen_info,
                       demand=demand,
                       wind_predict_power=wind_predict_power,
                       solar_predict_power=solar_predict_power,
                       SCHEDULE_LEN=24,
                       SEASON_FACTOR=0.5,
                       RESERVE_FACTOR=0.05,
                       ENS=10000,
                       NUM_THERMAL_GEN=4,
                       NUM_HYDRO_GEN=4)

    m.model()
    m.print_results()
    m.plot_figs()

    # If you wanna multi-solutions with certain variables, the following code may help you.
    # In this example, SEASON_FACTOR changes in [0.5, 0.7, 0.9].

    # SEASON_FACTOR_T = [0.5, 0.7, 0.9]
    # for SEASON_FACTOR in SEASON_FACTOR_T:
    #     m = UnitCommitment(thermal_gen_info=thermal_gen_info,
    #                        hydro_gen_info=hydro_gen_info,
    #                        demand=demand,
    #                        wind_predict_power=wind_predict_power,
    #                        solar_predict_power=solar_predict_power,
    #                        SCHEDULE_LEN=24,
    #                        SEASON_FACTOR=SEASON_FACTOR,
    #                        RESERVE_FACTOR=0.05,
    #                        ENS=10000,
    #                        NUM_THERMAL_GEN=4,
    #                        NUM_HYDRO_GEN=4)
    #
    #     m.model()
    #     m.print_results()
    #     m.plot_figs()


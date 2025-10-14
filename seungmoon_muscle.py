# Author(s): Seungmoon Song <seungmoon.song@gmail.com>
"""
used in:
- Song and Geyer. "Predictive neuromechanical simulations indicate why
walking performance declines with aging." The Journal of physiology,
2018.

adapted from:
- Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The
Journal of physiology, 2015.
- Geyer and Herr. "A muscle-reflex model that encodes
principles of legged mechanics produces human walking dynamics and muscle
activities." IEEE Transactions on neural systems and rehabilitation
engineering, 2010.
"""

from __future__ import division # '/' always means non-truncating division
import numpy as np


class MusculoTendonJoint(object):
    n = 0 # counter

    # f_lce
    W = .56
    C = np.log(.05)

    # f_vce
    N = 1.5
    K = 5

    # f_pe
    E_REF_PE = W
    # f_be
    E_REF_BE = .5*W
    E_REF_BE2 = 1 - W
    # f_se
    E_REF = .04

    # ECC
    TAU_ACT = .01 # [s]
    TAU_DACT = .04 # [s]

    # S range
    RNG_S = [.01, 1]


    def __init__(self, TIMESTEP, kw, n_update=1):
        MusculoTendonJoint.n += 1

        # simulation timestep
        self.TIMESTEP = TIMESTEP
        self.TIMESTEP_inter = TIMESTEP/n_update
        self.n_update = n_update

        # muscle parameters
        self.F_max = kw['F_max']
        self.l_opt = kw['l_opt']
        self.v_max = kw['v_max']
        self.l_slack = kw['l_slack']

        # muscle attachment parameters
        self.r1 = kw['r1'] # > 0 if l_mtu ~ phi1
        self.phi1_ref = kw['phi1_ref']
        self.rho = kw['rho']
        self.r2 = 0 if 'r2' not in kw else kw['r2'] # > 0 if l_mtu ~ phi2
        self.phi2_ref = 0 if 'phi2_ref' not in kw else kw['phi2_ref']

        self.A_init = .01 if 'A' not in kw else kw['A']

        phi1 = kw['phi1_0']
        phi2 = None if 'phi2_0' not in kw else kw['phi2_0']

        # -----------------------------------------------
        #angles from degrees to radians
        self.phi1_ref *= np.pi/180
        phi1 *= np.pi/180

        self.phi2_ref *= np.pi/180
        phi2 = phi2*np.pi/180 if phi2 is not None else None
        # -----------------------------------------------

        self.flag_afferent = 0 if 'flag_afferent' not in kw else kw['flag_afferent']
        if self.flag_afferent:
            self.del_t = kw['del_t'] # sensory delay

        self.name = 'muscle' + str(MusculoTendonJoint.n) if 'name' not in kw else kw['name']

        self.reset(phi1, phi2)

    def __del__(self):
        nn = "empty"

    def reset(self, phi1, phi2=None):
        self.phi1 = phi1
        self.phi2 = phi2

        self.v_ce = 0
        self.F_mtu = 0
        self.A = self.A_init
        self.S = self.A

        l_mtu = self.l_slack + self.l_opt
        l_mtu -= self.rho*self.r1*(self.phi1 - self.phi1_ref)
        if self.phi2 is not None:
            l_mtu -= self.rho*self.r2*(self.phi2 - self.phi2_ref)        
        self.l_ce = max(0.01, l_mtu - self.l_slack)  # 최소 길이 제한

        if self.flag_afferent:
            l_que = round(self.del_t/self.TIMESTEP)
            self.aff_l_ce0 = [self.l_ce/self.l_opt]*l_que
            self.aff_v_ce0 = [0]*l_que
            self.aff_F_mtu0 = [0]*l_que

    def update_inter(self, S, phi1, phi2=None, n_update=1):
        MTU = MusculoTendonJoint

        # ECC: update self.A
        self.fn_ECC(S)

        # calculate l_mtu
        l_mtu = self.l_slack + self.l_opt
        print(f"{l_mtu=}, {self.l_slack=}, {self.l_opt=}")
        l_mtu -= self.rho*self.r1*(phi1 - self.phi1_ref)
        print(f"{self.rho=}, {self.r1=}, {phi1=}, {self.phi1_ref=}")
        if phi2 is not None:
            l_mtu -= self.rho*self.r2*(phi2 - self.phi2_ref)

        # update muscle state
        self.l_se = l_mtu - self.l_ce# l_ce = l_mtu - l_slack -> l_se = l_slack - ...
        print(f"{l_mtu=}, {self.l_ce=}, {self.l_se=}")
        print(f"{self.l_se/self.l_slack=}")# This coudn't be more than 1 at the beginning
        f_se0 = fn_f_p0(self.l_se/self.l_slack, MTU.E_REF)
        f_be0 = fn_f_p0_ext(self.l_ce/self.l_opt, MTU.E_REF_BE, MTU.E_REF_BE2)
        f_pe0 = fn_f_p0(self.l_ce/self.l_opt, MTU.E_REF_PE)
        f_lce0 = fn_f_lce0(self.l_ce/self.l_opt, MTU.W, MTU.C)
        f_vce0 = (f_se0 + f_be0)/(f_pe0 + self.A*f_lce0)
        #f_vce0 = (f_se0 + f_be0 - f_pe0)/(self.A*f_lce0)
        v_ce0 = fn_inv_f_vce0(f_vce0, MTU.K, MTU.N)

        self.v_ce = self.l_opt*self.v_max*v_ce0
        self.l_ce = max(0.01, self.l_ce + self.v_ce*self.TIMESTEP_inter)  # 최소 길이 제한
        self.F_mtu = self.F_max*f_se0

    def update(self, S, phi1, phi2 = None):
        n_update = self.n_update
        if n_update == 1:
            self.update_inter(S, phi1, phi2)
        else:
            v_S = np.linspace(S, self.S, n_update, endpoint=False)
            v_phi1 = np.linspace(phi1, self.phi1, n_update, endpoint=False)
            v_ce_mean = 0
            l_ce_mean = 0
            F_mtu_mean = 0
            if phi2 is not None:
                v_phi2 = np.linspace(phi2, self.phi2, n_update, endpoint=False)
            for i in range(n_update):
                S_inter = v_S[-i-1]
                phi1_inter = v_phi1[-i-1]
                phi2_inter = None if phi2 is None else v_phi2[-i-1]
                self.update_inter(S_inter, phi1_inter, phi2_inter, n_update=n_update)
                v_ce_mean += self.v_ce/n_update
                l_ce_mean += self.l_ce/n_update
                F_mtu_mean += self.F_mtu/n_update
            self.v_ce = v_ce_mean
            self.l_ce = l_ce_mean
            self.F_mtu = F_mtu_mean

        if self.flag_afferent:
            self.updateSensor()

        self.S = S
        self.phi1 = phi1
        self.phi2 = phi2

    def getTorque(self):
        return self.F_mtu*self.r1

    def getTorque2(self):
        return self.F_mtu*self.r2

    def getSensoryData(self, s_data):
        if s_data == 'F_mtu':
            return self.aff_F_mtu0[0]
        elif s_data == 'l_ce':
            return self.aff_l_ce0[0]
        elif s_data == 'v_ce':
            return self.aff_v_ce0[0]
        else:
            raise ValueError('wrong data query!!')

    def updateSensor(self):
        self.aff_v_ce0.append(self.v_ce/(self.l_opt*self.v_max))
        self.aff_v_ce0.pop(0)
        self.aff_l_ce0.append(self.l_ce/self.l_opt)
        self.aff_l_ce0.pop(0)
        self.aff_F_mtu0.append(self.F_mtu/self.F_max)
        self.aff_F_mtu0.pop(0)

    def fn_ECC(self, S):
        MTU = MusculoTendonJoint

        S = np.clip(S, MTU.RNG_S[0], MTU.RNG_S[1])
        A = self.A
        if S > A:
            tau = MTU.TAU_ACT
        else:
            tau = MTU.TAU_DACT

        dA = (S - A)/tau
        self.A = A + dA*self.TIMESTEP_inter

        return self.A

    def getStates(self):
        F_mtu0 = self.F_mtu / self.F_max
        l_ce0 = self.l_ce / self.l_opt
        v_ce0 = self.v_ce / (self.l_opt*self.v_max)
        return (self.name, F_mtu0, l_ce0, v_ce0)


# MTU based on Geyer2010
def fn_inv_f_vce0(f_vce0, K, N):
    if f_vce0 <= 1:
        v_ce0 = (f_vce0 - 1)/(K*f_vce0 + 1)
    elif f_vce0 > 1 and f_vce0 <= N:
        temp = (f_vce0 - N)/(f_vce0 - N + 1)
        v_ce0 = (temp + 1)/(1 - 7.56*K*temp)
    else: # elif f_vce0 > N:
        v_ce0 = .01*(f_vce0 - N) + 1

    return v_ce0

# MTU based on Geyer2003
#def fn_inv_f_vce0(f_vce0, K, N):
#    if f_vce0 <= 1:
#        v_ce0 = (f_vce0 - 1)/(K*f_vce0 + 1)
#    elif f_vce0 > 1 and f_vce0 <= N:
#        v_ce0 = -(f_vce0 - 1)/((f_vce0-N)*7.56*K - (N-1))
#    else: # elif f_vce0 > N:
#        v_ce0 = .01*(f_vce0 - N) + 1
#
#    return v_ce0

def fn_f_lce0(l_ce0, w, c):
    f_lce0 = np.exp(c*np.abs((l_ce0-1)/(w))**3)
    return f_lce0

# f_p0 is for both f_se0 and f_pe0
def fn_f_p0(l0, e_ref):
    if l0 > 1:
        f_p0 = ((l0 - 1)/(e_ref))**2
    else:
        f_p0 = 0

    return f_p0

# for both f_be0
def fn_f_p0_ext(l0, e_ref, e_ref2):
    if l0 < e_ref2:
        f_p0 = ((l0 - e_ref2)/(e_ref))**2
    else:
        f_p0 = 0

    return f_p0


# =============================================================================
# Calculation Examples using MusculoTendonJoint
# =============================================================================

def single_step_calculation_example():
    """
    Single step calculation example using MusculoTendonJoint.
    This demonstrates how to use the muscle model for one time step.
    """
    print("=== Single Step Calculation Example ===")
    
    # Simulation parameters
    TIMESTEP = 0.001  # 1ms timestep
    
    # Muscle parameters (example values for a typical muscle)
    # 수정된 파라미터: 현실적인 근육 모델
    muscle_params = {
        'F_max': 1000.0,      # Maximum force [N]
        'l_opt': 0.2,          # Optimal muscle length [m] - 원래대로
        'v_max': 10.0,         # Maximum contraction velocity [m/s]
        'l_slack': 0.1,       # Slack tendon length [m] - 매우 작게
        'r1': 0.005,             # Moment arm [m] - 4배로 증가
        'phi1_ref': 0.0,       # Reference angle [deg]
        'rho': 1.0,            # Scaling factor
        'phi1_0': 0.0,         # Initial angle [deg] - 원래대로
        'A': 0.1,              # Initial activation
        'name': 'example_muscle'
    }
    
    # Create muscle instance
    muscle = MusculoTendonJoint(TIMESTEP, muscle_params)
    
    # Initial state
    # print(f"Initial muscle state:")
    # print(f"  Name: {muscle.name}")
    # print(f"  F_max: {muscle.F_max} N")
    # print(f"  l_opt: {muscle.l_opt} m")
    # print(f"  Initial activation: {muscle.A:.3f}")
    # print(f"  Initial l_ce: {muscle.l_ce:.4f} m")
    # print(f"  Initial F_mtu: {muscle.F_mtu:.2f} N")
    
    # Debug: Calculate MTU length and tendon length before update
    # phi1 = 0.0 * np.pi / 180
    # l_mtu = muscle.l_slack + muscle.l_opt
    # l_mtu -= muscle.rho * muscle.r1 * (phi1 - muscle.phi1_ref)
    # l_se = l_mtu - muscle.l_ce
    # l_se_ratio = l_se / muscle.l_slack
    
    # print(f"\n=== Debug Information ===")
    # print(f"  l_mtu: {l_mtu:.4f} m")
    # print(f"  l_se: {l_se:.4f} m")
    # print(f"  l_se/l_slack ratio: {l_se_ratio:.4f}")
    # print(f"  E_REF: {MusculoTendonJoint.E_REF}")
    
    # Single step calculation
    S = 0.5  # Neural activation signal
    phi1 = 0.1 * np.pi / 180  # 원래 각도로 복원
    
    # print(f"\nSingle step update:")
    # print(f"  Input S: {S}")
    # print(f"  Input phi1: {phi1 * 180 / np.pi:.1f} deg")
    
    # Update muscle state
    for idx in range(10):
        print(f"{'='*10}{idx=}{'='*10}")
        muscle.update(S, phi1)
    
    # Results
    print(f"\nResults after single step:")
    print(f"  Activation A: {muscle.A:.3f}")
    print(f"  Contractile element length l_ce: {muscle.l_ce:.4f} m")
    print(f"  Contractile element velocity v_ce: {muscle.v_ce:.4f} m/s")
    print(f"  Muscle-tendon unit force F_mtu: {muscle.F_mtu:.2f} N")
    print(f"  Torque: {muscle.getTorque():.2f} Nm")
    
    # Debug: Show intermediate calculations
    print(f"\n=== Debug: Intermediate Calculations ===")
    print(f"  l_se after update: {muscle.l_se:.4f} m")
    print(f"  l_se/l_slack: {muscle.l_se/muscle.l_slack:.4f}")
    
    return muscle


def multistep_calculation_example():
    """
    Multistep calculation example using MusculoTendonJoint.
    This demonstrates how to use the muscle model for multiple time steps
    with different update frequencies.
    """
    print("\n=== Multistep Calculation Example ===")
    
    # Simulation parameters
    TIMESTEP = 0.01  # 10ms timestep
    n_update = 10   # Number of internal updates per timestep
    
    # Muscle parameters
    muscle_params = {
        'F_max': 2000.0,      # Maximum force [N]
        'l_opt': 0.12,         # Optimal muscle length [m]
        'v_max': 8.0,          # Maximum contraction velocity [m/s]
        'l_slack': 0.08,       # Slack tendon length [m]
        'r1': 0.08,            # Moment arm [m]
        'r2': 0.06,            # Second moment arm [m]
        'phi1_ref': 0.0,       # Reference angle 1 [deg]
        'phi2_ref': 0.0,       # Reference angle 2 [deg]
        'rho': 1.0,            # Scaling factor
        'phi1_0': 0.0,         # Initial angle 1 [deg]
        'phi2_0': 0.0,         # Initial angle 2 [deg]
        'A': 0.05,             # Initial activation
        'name': 'multistep_muscle'
    }
    
    # Create muscle instance with multistep updates
    muscle = MusculoTendonJoint(TIMESTEP, muscle_params, n_update=n_update)
    
    print(f"Multistep muscle configuration:")
    print(f"  Name: {muscle.name}")
    print(f"  Timestep: {TIMESTEP} s")
    print(f"  Internal updates per step: {n_update}")
    print(f"  Internal timestep: {muscle.TIMESTEP_inter:.6f} s")
    
    # Simulation parameters
    n_steps = 100
    time_points = np.linspace(0, n_steps * TIMESTEP, n_steps)
    
    # Create time-varying inputs
    S_signal = 0.3 + 0.2 * np.sin(2 * np.pi * 0.5 * time_points)  # Neural activation
    phi1_signal = 45 * np.pi / 180 * np.sin(2 * np.pi * 0.1 * time_points)  # Joint angle 1
    phi2_signal = 30 * np.pi / 180 * np.cos(2 * np.pi * 0.1 * time_points)  # Joint angle 2
    
    # Storage arrays
    results = {
        'time': time_points,
        'S': S_signal,
        'phi1': phi1_signal,
        'phi2': phi2_signal,
        'A': np.zeros(n_steps),
        'l_ce': np.zeros(n_steps),
        'v_ce': np.zeros(n_steps),
        'F_mtu': np.zeros(n_steps),
        'torque1': np.zeros(n_steps),
        'torque2': np.zeros(n_steps)
    }
    
    print(f"\nRunning {n_steps} simulation steps...")
    
    # Run simulation
    for i in range(n_steps):
        # Update muscle state
        muscle.update(S_signal[i], phi1_signal[i], phi2_signal[i])
        
        # Store results
        results['A'][i] = muscle.A
        results['l_ce'][i] = muscle.l_ce
        results['v_ce'][i] = muscle.v_ce
        results['F_mtu'][i] = muscle.F_mtu
        results['torque1'][i] = muscle.getTorque()
        results['torque2'][i] = muscle.getTorque2()
    
    # Print summary statistics
    print(f"\nSimulation completed!")
    print(f"  Final activation: {results['A'][-1]:.3f}")
    print(f"  Final l_ce: {results['l_ce'][-1]:.4f} m")
    print(f"  Final v_ce: {results['v_ce'][-1]:.4f} m/s")
    print(f"  Final F_mtu: {results['F_mtu'][-1]:.2f} N")
    print(f"  Final torque1: {results['torque1'][-1]:.2f} Nm")
    print(f"  Final torque2: {results['torque2'][-1]:.2f} Nm")
    
    print(f"\nStatistics:")
    print(f"  Max activation: {np.max(results['A']):.3f}")
    print(f"  Max force: {np.max(results['F_mtu']):.2f} N")
    print(f"  Max torque1: {np.max(results['torque1']):.2f} Nm")
    print(f"  Max torque2: {np.max(results['torque2']):.2f} Nm")
    
    return muscle, results


def compare_single_vs_multistep():
    """
    Compare single step vs multistep calculation accuracy.
    """
    print("\n=== Single vs Multistep Comparison ===")
    
    TIMESTEP = 0.01
    
    # Common muscle parameters - 수정된 파라미터
    base_params = {
        'F_max': 1500.0,
        'l_opt': 0.1,
        'v_max': 10.0,
        'l_slack': 0.15,        # 매우 큰 slack length
        'r1': 0.005,            # 매우 작은 moment arm
        'phi1_ref': 0.0,
        'rho': 1.0,
        'phi1_0': 0.0,
        'A': 0.1,
        'name': 'comparison_muscle'
    }
    
    # Create single step muscle
    muscle_single = MusculoTendonJoint(TIMESTEP, base_params.copy(), n_update=1)
    muscle_single.name = 'single_step'
    
    # Create multistep muscle
    muscle_multistep = MusculoTendonJoint(TIMESTEP, base_params.copy(), n_update=10)
    muscle_multistep.name = 'multistep'
    
    # Test with same inputs
    S = 0.6
    phi1 = 0 * np.pi / 180  # 더 작은 각도로 변경
    
    print(f"Testing with S={S}, phi1={phi1*180/np.pi:.1f} deg")
    print(f"Timestep: {TIMESTEP} s")
    print(f"Single step internal timestep: {muscle_single.TIMESTEP_inter:.6f} s")
    print(f"Multistep internal timestep: {muscle_multistep.TIMESTEP_inter:.6f} s")
    
    # Single step calculation
    muscle_single.update(S, phi1)
    
    # Multistep calculation
    muscle_multistep.update(S, phi1)
    
    print(f"\nResults comparison:")
    print(f"  Single step - A: {muscle_single.A:.6f}, F_mtu: {muscle_single.F_mtu:.2f} N")
    print(f"  Multistep   - A: {muscle_multistep.A:.6f}, F_mtu: {muscle_multistep.F_mtu:.2f} N")
    print(f"  Difference  - A: {abs(muscle_single.A - muscle_multistep.A):.6f}, F_mtu: {abs(muscle_single.F_mtu - muscle_multistep.F_mtu):.2f} N")
    
    # Additional detailed comparison
    print(f"\nDetailed comparison:")
    print(f"  Single step - l_ce: {muscle_single.l_ce:.6f} m, v_ce: {muscle_single.v_ce:.6f} m/s")
    print(f"  Multistep   - l_ce: {muscle_multistep.l_ce:.6f} m, v_ce: {muscle_multistep.v_ce:.6f} m/s")
    print(f"  Difference  - l_ce: {abs(muscle_single.l_ce - muscle_multistep.l_ce):.6f} m, v_ce: {abs(muscle_single.v_ce - muscle_multistep.v_ce):.6f} m/s")
    
    # Calculate relative differences
    rel_diff_A = abs(muscle_single.A - muscle_multistep.A) / muscle_single.A * 100
    rel_diff_F = abs(muscle_single.F_mtu - muscle_multistep.F_mtu) / max(muscle_single.F_mtu, 1e-6) * 100
    print(f"\nRelative differences:")
    print(f"  Activation A: {rel_diff_A:.2f}%")
    print(f"  Force F_mtu: {rel_diff_F:.2f}%")


if __name__ == "__main__":
    """
    Run the calculation examples when the script is executed directly.
    """
    print("MusculoTendonJoint Calculation Examples")
    print("=" * 50)
    
    # Run single step example
    muscle1 = single_step_calculation_example()
    
    # Run multistep example
    # muscle2, results = multistep_calculation_example()
    
    # Compare single vs multistep
    # compare_single_vs_multistep()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
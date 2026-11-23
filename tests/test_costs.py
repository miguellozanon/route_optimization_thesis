from brlo.data.instances import iberia_base
from brlo.data.model import Params
from brlo.longhaul.costs import hos_schedule, leg_loads, trip_cost_model


def test_hos_short_trip_no_rest():
    t = hos_schedule([2.0, 2.0], [1.0], Params())
    assert t.rests == 0 and t.breaks_h == 0 and abs(t.elapsed_h - 5.0) < 1e-9


def test_hos_break_after_4h30():
    t = hos_schedule([5.0], [], Params())
    assert t.breaks_h == 0.75 and t.rests == 0


def test_hos_daily_rest_after_9h():
    t = hos_schedule([10.0], [], Params())
    # 4.5 drive + 0.75 break + 4.5 drive -> 9 h driving -> 11 h rest -> 1 h drive
    assert t.rests == 1 and abs(t.elapsed_h - (10 + 0.75 + 11)) < 1e-9


def test_trip_cost_linear_in_load():
    inst = iberia_base()
    v = inst.V["T40"]
    m = trip_cost_model(inst, "MAD", v, ("SNAM_VIV", "ACC_KRE"))
    loads = leg_loads(m.stops, {(0, "NMC"): 5000, (1, "NMC"): 7000}, tare=1000)
    assert loads == [13000, 8000, 1000]
    assert all(b > 0 for b in m.beta)
    assert m.time.rests >= 3          # ~3,700 km cannot be driven in less than 4 duty days

import random


class TrafficJunction:

    def __init__(self, name):
        self.name = name
        self.queue = random.randint(5, 15)

    def get_state(self):
        if self.queue <= 5:
            return "LOW"
        elif self.queue <= 12:
            return "MEDIUM"
        else:
            return "HIGH"

    def add_vehicles(self):
        arrivals = random.randint(1, 5)
        self.queue += arrivals

    def allow_vehicles(self, signal_duration):

        if signal_duration == 10:
            vehicles_passed = 3

        elif signal_duration == 20:
            vehicles_passed = 6

        else:
            vehicles_passed = 9

        vehicles_passed = min(vehicles_passed, self.queue)

        self.queue -= vehicles_passed

        return vehicles_passed
    
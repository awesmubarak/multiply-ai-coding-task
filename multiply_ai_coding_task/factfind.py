import datetime as dt
from dataclasses import dataclass
from enum import Enum


class GoalType(Enum):
    NEW_HOME = "new_home"
    NEW_CAR = "new_car"
    OTHER = "other"


@dataclass
class NewHomeGoalInformation:
    location: str
    house_price: float
    deposit_amount: float
    purchase_date: dt.date


@dataclass
class NewCarInformation:
    car_type: str
    car_price: float
    purchase_date: dt.date


@dataclass
class OtherGoalInformation:
    description: str
    amount_required: float
    target_date: dt.date


@dataclass
class Goal:
    goal_type: GoalType
    goal_name: str
    goal_specific_information: (
        NewHomeGoalInformation | NewCarInformation | OtherGoalInformation
    )


@dataclass
class User:
    first_name: str
    last_name: str
    email: str
    date_of_birth: dt.date

    goals: list[Goal]

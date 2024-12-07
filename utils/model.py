import numpy as np
import pandas as pd
import pickle

class Model:
    def __init__(self, params={}):
        self.params = params

    def train(self, X, y):
        pass

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X, y_labels):
        assert len(X) == len(y_labels), 'X and y must have the same length'
        y_preds = self.predict(X)
        return 1 - np.mean(abs(y_preds - y_labels))
    
    @staticmethod
    def process_sex(series, dict = {'male': -1, 'female': 1}):
        return series.map(dict).astype(int)

        

class ConstantBaselineModel(Model):
    def __init__(self, params={'survived': False}):
        super().__init__(params)

    def predict(self, X):
        return np.zeros(X.shape[0]) if not self.params['survived'] else np.ones(X.shape[0])
    

class SexBaselineModel(Model):
    def __init__(self, params={}):
        super().__init__(params)

    def predict(self, X):
        sex = self.process_sex(X['Sex'])
        return ((sex + 1) / 2).astype(int)
    
class TicketUponLifeboatModel(Model):
    ticket_survival_distribution = pd.read_csv('../data/ticket_survival_rates.csv', index_col=0)

    def __init__(self, params={}):
        super().__init__(params)

    def predict(self, X):
        X = self.apply_ticket_statistics_to_data(X)
        return X.apply(lambda row: int(row['TicketSurvivalLikelihood'] > 0.5) if row['TicketSize'] < 4 else int(row['Sex'] == 'female'), axis=1)

    def apply_ticket_statistics_to_data(self, X):
        ticket_size_grouping = pd.DataFrame(X.groupby('Ticket').size(), columns=['TestTicketSize'])

        # Get the number of people in the training data on the ticket
        ticket_size_grouping = ticket_size_grouping.merge(self.load_training_data_ticket(X), how='left', left_index=True, right_index=True)
        for col in ticket_size_grouping.columns:
            ticket_size_grouping[col] = ticket_size_grouping[col].fillna(0).astype(int)
        
        # Get the marginal likelihood of one (more) passenger on this ticket surviving
        p_n_tuples = ticket_size_grouping.apply(TicketUponLifeboatModel.generate_ticket_statistics, axis=1)

        ticket_size_grouping['TicketSurvivalLikelihood'], ticket_size_grouping['TicketSize'] = p_n_tuples.apply(lambda x: x[0]), p_n_tuples.apply(lambda x: x[1])
        ticket_size_grouping['TicketSurvivalLikelihood'] = ticket_size_grouping['TicketSurvivalLikelihood'].astype(float)
        ticket_size_grouping['TicketSize'] = ticket_size_grouping['TicketSize'].fillna(0).astype(int)
        X = X.merge(ticket_size_grouping[['TicketSurvivalLikelihood', 'TicketSize']], how='left', left_on='Ticket', right_index=True)
        return X

    def load_training_data_ticket(self, X) -> pd.DataFrame:
        '''Load the training data and compute the number of people on the associated ticket
        
        return: pd.DataFrame grouped by ticket with the current number of training data survivors and training passengers
        
        '''
        train = pd.read_csv('../data/train.csv')

        # We'll avoid counting  the passengers in the X, 
        # so that we don't double count passengers on a ticket during prediction.
        # We can also ignore counting those tickets not in X,
        # as these won't be used during prediction.
        pred_tickets = set(X['Ticket'].to_list())
        pred_passengers = set(X['PassengerId'].to_list())
        train = train[(~train.PassengerId.isin(pred_passengers)) & (train.Ticket.isin(pred_tickets))]

        return train.groupby('Ticket').agg(
            {'Survived': 'sum',
             'PassengerId': 'count'}).rename(
                 columns={'PassengerId': 'TrainingTicketSize', 'Survived': 'TrainingSurvived'}
                 )
    @staticmethod
    def generate_ticket_statistics(row):
        m = row['TestTicketSize'] # Int, at least 1
        n = row['TrainingTicketSize'] + m # Int, at least m
        # Return NaN if n > 7, as we have no training data for this case
        if n > 5:
            return np.nan, n
        k = row['TrainingSurvived'] # Int, at least 0

        M = TicketUponLifeboatModel.ticket_survival_distribution.loc[n][k:k+m+1].values
        p = 1 - (M[0]/M.sum())
        return p, n
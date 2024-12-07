import pandas as pd
import pickle

def get_train_test_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    return train, test

def get_tickets_set(df):
    return set(df['Ticket'].to_list())

def main():
    train, test = get_train_test_data()
    train_tickets = get_tickets_set(train)
    test_tickets = get_tickets_set(test)
    tickets_in_both = train_tickets.intersection(test_tickets)
    with open('../data/tickets_in_train_and_test.pkl', 'wb') as f:
        pickle.dump(tickets_in_both, f)
    with open('../data/tickets_in_train.pkl', 'wb') as f:
        pickle.dump(train_tickets, f)


if __name__ == '__main__':
    main()
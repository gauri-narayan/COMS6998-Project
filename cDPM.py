import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F

import pyro
from pyro.distributions import *
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam

assert pyro.__version__.startswith('1.5.0')
pyro.enable_validation(True)       # can help with debugging
pyro.set_rng_seed(0)

from get_DPM_input_rep import *

##t=0; DPM

# "stick breaking function"
def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

def model_t0(data):
    with pyro.plate("beta_plate", T - 1):
        beta = pyro.sample("beta", Beta(1, alpha))  # same as example #alpha is hyperparameter, scaling

    with pyro.plate("theta_plate", T):  # shape [T,C]
        # sample probabilities for Mult dist.; Dirichlet distribution (conjugate prior of Mult); symmetric prior
        theta = pyro.sample("theta", Dirichlet(torch.ones(C) / C))

    with pyro.plate("data", N):
        # z==which topic
        z = pyro.sample("z", Categorical(probs=mix_weights(beta)))  # same as example; #[N]
        pyro.sample("obs", Multinomial(probs=theta[z]), obs=data)  # x_i | theta_i ~ Mult(\theta_i)


def guide_t0(data):
    # T-1 alpha params for beta sampling
    kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T - 1]), constraint=constraints.positive)

    # concentration params for q_theta #[T,C]
    tau = pyro.param('tau', lambda: MultivariateNormal(0.5 * torch.ones(C), 0.25 * torch.eye(C)).sample([T]),
                     constraint=constraints.unit_interval)

    # N params for categorical dist; topic weights; symmetric prior
    phi = pyro.param('phi', lambda: Dirichlet(1 / T * torch.ones(T)).sample([N]), constraint=constraints.simplex)

    with pyro.plate("beta_plate", T - 1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T - 1), kappa))

    # sample probs for multinomial distributions
    with pyro.plate("theta_plate", T):
        q_theta = pyro.sample("theta", Dirichlet(tau))  # outputs multinomial probabilities for each topic

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(phi))


def truncate(alpha, probs, taus, weights):
    threshold = alpha ** -1 / 100.
    true_weights = weights[weights > threshold] / torch.sum(weights[weights > threshold])
    true_probs = probs[weights > threshold]
    true_taus = taus[weights>threshold]
    return true_probs, true_weights, true_taus


def get_topic_assignments_t0(data, tau_final, topic_weights):
    # get expected multinomial parameters based on Dirichlet distribution
    #topic_weights = mix_weights(1. / (1. + kappa_final))
    #print("theta_exp")
    theta_exp = []
    for i in range(0, T):
        ratio = tau_final[i] / sum(tau_final[i])
        theta_exp.append(ratio)
    theta_exp = torch.stack(theta_exp)
    #print(theta_exp)
    # get pruned topics+weights
    true_probs, true_weights, true_taus = truncate(alpha, theta_exp, tau_final, topic_weights)
    topic_assignments = []

    # topic assignment
    old_topics = []
    for x in data:
        max_prob = 0
        max_prob_topic = 0
        for t in range(0, len(true_weights)):
            t_weight = true_weights[t]
            dist = Multinomial(probs=true_probs[t])
            # likelihood of being from topic t
            log_prob = -np.log(t_weight) - dist.log_prob(x)
            if log_prob >= max_prob:
                max_prob = log_prob
                max_prob_topic = t

        topic_assignments.append(max_prob_topic)
    #T_total+=len(true_weights)
    return torch.tensor(topic_assignments), true_probs, true_taus


# DPM t>=1
def model(data):
    # whether new topic or not; prior=0.5; random choice whether old/new
    with pyro.plate("new_topic_plate", T):
        new_topic = pyro.sample("new_topic", Binomial(probs=0.5))

    # if new topic, if linked to old topic, prior=0.5
    with pyro.plate("linked_plate", T):
        linked = pyro.sample("linked", Binomial(probs=0.5))

    # if old topic, which old topic; prior = frequencies of topics in previous days
    with pyro.plate("old_topic_plate", T):
        which_old_topic = pyro.sample("which_old_topic", Multinomial(probs=prev_topic_freq))

    # beta sampling for topic weights
    with pyro.plate("beta_plate", T - 1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("theta_plate", T):  # shape [T,C]
        # sample probabilities for Mult dist.; Dirichlet distribution (conjugate prior of Mult); symmetric prior
        theta = pyro.sample("theta", Dirichlet(torch.ones(C) / C))

    with pyro.plate("gamma_plate", T_prev):
        gamma = pyro.sample("gamma", Dirichlet(prev_taus))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(probs=mix_weights(beta)))  # same as example; #[N]
        #which_old_topic = torch.tensor(which_old_topic, dtype=bool)
        old = get_old_topics(which_old_topic)
        a = (new_topic) * (linked)
        b = (1 - new_topic)
        c = (new_topic) * (1 - linked)
        a = a[z].reshape(N, 1)
        b = b[z].reshape(N, 1)
        c = c[z].reshape(N, 1)
        mult_probs = a * gamma[old[z]] + b * prev_theta[old[z]] + c * theta[z]
        pyro.sample("obs", Multinomial(probs=mult_probs), obs=data)  # x_i | theta_i ~ Mult(\theta_i)


def guide(data):
    ##pyro params:
    new_topic_prob = pyro.param("new_topic_prob", lambda: Uniform(0, 1).sample([T]),
                                constraint=constraints.unit_interval)

    linked_prob = pyro.param("linked_prob", lambda: Uniform(0, 1).sample([T]), constraint=constraints.unit_interval)

    which_topic_probs = pyro.param("which_topic_probs", lambda: Uniform(0, 1).sample([T_prev]),constraint=constraints.simplex)

    kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T - 1]), constraint=constraints.positive)
    # print(kappa)

    tau = pyro.param('tau', lambda: MultivariateNormal(0.5 * torch.ones(C), 0.25 * torch.eye(C)).sample([T]),
                     constraint=constraints.unit_interval)

    # N params for categorical dist; topic weights; symmetric prior
    phi = pyro.param('phi', lambda: Dirichlet(1 / T * torch.ones(T)).sample([N]), constraint=constraints.simplex)

    # model params
    with pyro.plate("new_topic_plate", T):
        # print(new_topic_prob)
        new_topic = pyro.sample("new_topic", Binomial(probs=new_topic_prob))

    # if new topic, if linked to old topic, prior=0.5
    with pyro.plate("linked_plate", T):
        linked = pyro.sample("linked", Binomial(probs=linked_prob))

    # if old topic, which old topic
    with pyro.plate("old_topic_plate", T):
        which_old_topic = pyro.sample("which_old_topic", Multinomial(probs=which_topic_probs))

    with pyro.plate("beta_plate", T - 1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T - 1), kappa))

    # new topic with symmetric prior
    with pyro.plate("theta_plate", T):
        theta = pyro.sample("theta", Dirichlet(tau))

    # new topic linked to old topic
    with pyro.plate("gamma_plate", T_prev):
        #print(prev_taus)
        gamma = pyro.sample("gamma", Dirichlet(prev_taus))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(phi))
        #which_old_topic = torch.tensor(which_old_topic, dtype=bool)
        old = get_old_topics(which_old_topic)
        a = ((new_topic) * (linked))
        b = (1 - new_topic)
        c = ((new_topic) * (1 - linked))
        a = a[z].reshape(N,1)
        b = b[z].reshape(N,1)
        c = c[z].reshape(N,1)
        mult_probs = a * gamma[old[z]] + b * prev_theta[old[z]] + c * theta[z]


def get_old_topics(which_old_topic):
    ind = []
    for i in which_old_topic:
        ind.append(torch.where(i == 1)[0][0])
    ind = torch.stack(ind)
    return ind

def get_equiv_topics(new_topic_prob, which_topic_probs):
    new_to_old_topics = dict()
    for t in range(0, len(new_topic_prob)):
        if new_topic_prob[t]<0.5:
            old_topic = np.argmax(which_topic_probs)
            #print(old_topic)
            new_to_old_topics[t+T_total] = int(old_topic-T_prev+T_total)
    #print(new_to_old_topics)
    return new_to_old_topics



#assign each trial in the data a topic
def get_topic_assignments(data, tau_final, topic_weights, new_topic_prob, linked_prob, which_topic_probs,
                          old_mult_probs):
    # get expected multinomial parameters based on Dirichlet distribution
    theta_exp = []
    for i in range(0, T):
        ratio = tau_final[i] / sum(tau_final[i])
        theta_exp.append(ratio)
    theta_exp = torch.stack(theta_exp)

    # get pruned topics+weights
    true_probs, true_weights, true_taus = truncate(alpha, theta_exp, tau_final, topic_weights)
    topic_assignments = []
    old_new_equivalents = get_equiv_topics(new_topic_prob, which_topic_probs)
    for x in data:
        max_prob = 0
        max_prob_topic = 0
        #print(len(true_weights))
        for t in range(0, len(true_weights)):
            t_weight = true_weights[t]

            # likelihood of being from topic t
            log_prob = -1 * np.log(t_weight)  # -dist.log_prob(x)
            # likelihood of new topic
            new_old_probs = [new_topic_prob[t], 1 - new_topic_prob[t]]
            linked_new_probs = [linked_prob[t], 1 - linked_prob[t]]
            for n in range(0, 2):
                log_prob -= (-1) * (np.log(new_old_probs[n]))  # log prob of new topic
                for l in range(0, 2):
                    if n == 1:  # old_topic; use old_probs; link var doesn't matter
                        for p in range(0, len(which_topic_probs)):
                            log_prob -= np.log(which_topic_probs[p])  # probability of topic t==topic p
                            dist = Multinomial(probs=old_mult_probs[p])
                            log_prob -= dist.log_prob(x)
                    else:  # new topic
                        #whether linked or not, use theta_exp
                        log_prob -= np.log(linked_prob[l])
                        dist = Multinomial(probs=theta_exp[t])
                        log_prob -= dist.log_prob(x)

            if log_prob > max_prob:
                max_prob = log_prob
                max_prob_topic = t+T_total

        if max_prob_topic in old_new_equivalents:
            topic_assignments.append(old_new_equivalents[max_prob_topic])
        else:
            topic_assignments.append(max_prob_topic)
    return torch.tensor(topic_assignments), true_probs, true_taus

def train(num_iterations, svi):
    pyro.clear_param_store()
    for j in tqdm(range(num_iterations)):
        loss = svi.step(data)
        losses.append(loss)

#for each tweet, assign as its topic the mode of its word assignments
def get_tweet_assignments(bow, assignments):
    start = int(0)
    tweet_topic = np.ndarray(len(bow))
    i = 0
    for b in bow:
        num_trials = int(np.sum(b))
        word_assignments = assignments[start:start+num_trials]
        topic = torch.mode(word_assignments)[0]
        tweet_topic[i] = topic
        start+=num_trials
        i+=1
    return tweet_topic


def run_cDPM():
    # probs1 = torch.tensor([0.3, 0.7, 0])
    # probs2 = torch.tensor([0.8, 0.1, 0.1])
    # probs3 = torch.tensor([0.2, 0, 0.8])
    global data, N, C, T, alpha, losses, T_total, prev_taus, prev_theta, T_prev, prev_topic_freq, tweet_assignments
    # data = torch.cat((Multinomial(probs=probs1).sample([100]),
    #                    Multinomial(probs=probs2).sample([100]),
    #                    Multinomial(probs=probs3).sample([100])))
    T = 8
    optim = Adam({"lr": 0.01})
    alpha = 0.1
    T_total = 0
    all_companies_bow, all_companies_mult_inputs, all_companies_tweets = get_inputs()
    for company in all_companies_bow:
        #print(company)
        tweet_assignments = dict()
        first_day = True
        co_df = pd.DataFrame(columns=["date", "tweet", "topic"])
        i=0
        for date in all_companies_mult_inputs[company]:
            print(date)
            #if i==1:
                #break
            i+=1
            #bow = all_companies_bow[date]
            data = np.array(all_companies_mult_inputs[company][date])
            data = torch.tensor(data)
            #print(data)
            N = data.shape[0]
            C = data.shape[1]
            #print(data.shape)
            if first_day: #use model_t0
                losses = []
                svi_t0 = SVI(model_t0, guide_t0, optim, loss=Trace_ELBO())
                train(1000, svi_t0)
                tau_final = pyro.param("tau").detach()
                kappa_final = pyro.param("kappa").detach()
                topic_weights = torch.mean(pyro.param("phi").detach(), dim=0)
                assignments, prev_theta, prev_taus = get_topic_assignments_t0(data, tau_final, topic_weights)  # topic_weights)
                T_total+=len(prev_theta)
                tweet_assignments[date] = get_tweet_assignments(all_companies_bow[company][date], assignments)
                first_day=False
            else: #use sequential model
                T_prev = len(prev_theta)
                prev_topic_freq = torch.ones([T_prev]) * (1. / T_prev)
                svi_t1 = SVI(model, guide, optim, loss=Trace_ELBO())
                losses = []
                train(100, svi_t1)
                tau_final = pyro.param("tau").detach()
                topic_weights = torch.mean(pyro.param("phi").detach(), dim=0)
                new_topic_prob = pyro.param("new_topic_prob").detach()
                linked_prob = pyro.param("linked_prob").detach()
                which_topic_probs = pyro.param("which_topic_probs").detach()
                tau = pyro.param('tau').detach()
                topic_weights = torch.mean(pyro.param('phi').detach(), dim=0)
                assignments, prev_theta, prev_taus = get_topic_assignments(data, tau_final, topic_weights,
                                                                           new_topic_prob, linked_prob,
                                                                           which_topic_probs, prev_theta)
                tweet_assignments[date] = get_tweet_assignments(all_companies_bow[company][date], assignments)
                T_total+=len(prev_taus)
        return tweet_assignments, all_companies_tweets[company], company

if __name__ == "__main__":
    run_cDPM()


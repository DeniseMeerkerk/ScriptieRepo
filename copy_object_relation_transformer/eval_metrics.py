#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:39:22 2022

@author: denise
"""
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk

#nltk.download('omw-1.4')
nltk.download('wordnet')



#reference = "The lungs are hyperexpanded. There are stable scattered XXXX bilateral opacities, most notable in the left upper lobe, XXXX scarring. No focal airspace consolidation to suggest pneumonia. No large pleural effusion. No pneumothorax. Heart size is normal. Thoracic aorta is mildly tortuous and demonstrates atherosclerotic vascular calcification. There are degenerative changes of the spine."
#candidate = "the heart and lungs have xxxx in the interval both lungs are clear and expanded heart and mediastinum normal no active disease"

#reference_token = [reference.split()]
#candidate_token = candidate.split()
#%% class
class Eval_Metrics:
    def __init__(self,n_bleu=4):
        self.result = {}
        self.n_bleu = n_bleu
        self.result["METEOR"] = []
        self.result["rougeL_fmeasure"] = []
        self.result["CIDEr"] = []

        for i in range(self.n_bleu):
            self.result["BLEU"+str(i+1)] = []
        return
    
    def all_metrics(self,reference,candidate,metric_list=['METEOR',"ROUGE","BLEU","CIDER"]):
        if "METEOR" in metric_list:
            self.meteor(reference,candidate)
        if "BLEU" in metric_list:
            self.bleu(reference,candidate)
        if "ROUGE" in metric_list:
            self.rouge(reference,candidate)
        if "CIDER" in metric_list:
            self.cider(reference,candidate)
        
        return
        
        
    def bleu(self, reference, candidate):
        reference_token = [reference.split()]
        candidate_token = candidate.split()
        #TODO: WEIGHTS BUILD FROM SELF.N_BLEU
        weights = [(1., 0., 0., 0.),
                   (0., 1., 0., 0.),
                   (0., 0., 1., 0.),
                   (0., 0., 0., 1.)]
        for i in range(self.n_bleu):
            score_bleu = sentence_bleu(reference_token, candidate_token,weights[i])
            #print('BLEU score -> {}'.format(sentence_bleu(reference_token, candidate_token,weights[i])))
            self.result["BLEU"+str(i+1)].append(score_bleu*100)
        return
    
    def meteor(self, reference, candidate):
        reference_token = reference #.split() #for other versions of python/nltk this needs to be in a list
        candidate_token = candidate #.split()
        
        self.result["METEOR"].append(meteor_score(reference_token,candidate_token)*100)
        return
    
    def rouge(self, reference, candidate):
        rouge_scores = rouge_score_lcs(reference.split(),candidate.split())
        self.result["rougeL_fmeasure"].append(rouge_scores[-1]*100)
        return

    def cider(self, reference, candidate):
        #TODO
        self.result["CIDEr"].append(np.nan)
        return

from rouge_score import scoring


def rouge_score_lcs(target_tokens, prediction_tokens):
  """Computes LCS (Longest Common Subsequence) rouge scores.

  Args:
    target_tokens: Tokens from the target text.
    prediction_tokens: Tokens from the predicted text.
  Returns:
    A Score object containing computed scores.
  """

  if not target_tokens or not prediction_tokens:
    return scoring.Score(precision=0, recall=0, fmeasure=0)

  # Compute length of LCS from the bottom up in a table (DP appproach).
  lcs_table = _lcs_table(target_tokens, prediction_tokens)
  lcs_length = lcs_table[-1][-1]

  precision = lcs_length / len(prediction_tokens)
  recall = lcs_length / len(target_tokens)
  fmeasure = scoring.fmeasure(precision, recall)

  return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)

def _lcs_table(ref, can):
  """Create 2-d LCS score table."""
  rows = len(ref)
  cols = len(can)
  lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
  for i in range(1, rows + 1):
    for j in range(1, cols + 1):
      if ref[i - 1] == can[j - 1]:
        lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
      else:
        lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
  return lcs_table
#%%
#eval_metrics = Eval_Metrics(4)
#eval_metrics.all_metrics(reference,candidate)
#print(eval_metrics.result)

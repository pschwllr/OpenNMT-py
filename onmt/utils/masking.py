from __future__ import division
import torch
import numpy as np


class ChemVocabMasker(object):
    def __init__(self, vocab=None, from_file=None):

        if from_file is not None:
            checkpoint = torch.load(from_file)
            self.always_active = checkpoint['always_active']
            self.atom_vocab_dict = checkpoint['atom_vocab_dict']
            self.vocab_atom_dict = checkpoint['vocab_atom_dict']
            self.vocab_vocab_dict = checkpoint['vocab_vocab_dict']
            self.vocab = checkpoint['vocab']

        elif vocab is not None:
            self.vocab = vocab
            self.initialise_dicts()

        else:
            print("ChemVocabMasker: Not initialised.. ")

    def save_dicts(self, file_path):
        torch.save({
            'always_active': self.always_active,
            'atom_vocab_dict': self.atom_vocab_dict,
            'vocab_atom_dict': self.vocab_vocab_dict,
            'vocab_vocab_dict': self.vocab_vocab_dict,
            'vocab': self.vocab
        }, file_path)

    def initialise_dicts(self):
        from rdkit import Chem
        always_active = []
        atom_vocab_dict = {}
        for i, v in enumerate(self.vocab.itos):
            mol = Chem.MolFromSmiles(v)
            if mol is not None:
                atomic_num = mol.GetAtoms()[0].GetAtomicNum()

                if atomic_num in atom_vocab_dict.keys():
                    atom_vocab_dict[atomic_num].append(i)
                else:
                    atom_vocab_dict[atomic_num] = [i]
            else:
                new_v = ''
                first_alpha = True
                for c in v:
                    if first_alpha and c.isalpha():
                        new_v += c.upper()
                        first_alpha = False
                    else:
                        new_v += c
                mol = Chem.MolFromSmiles(new_v)

                if mol is not None:
                    atomic_num = mol.GetAtoms()[0].GetAtomicNum()

                    if atomic_num in atom_vocab_dict.keys():
                        atom_vocab_dict[atomic_num].append(i)
                    else:
                        atom_vocab_dict[atomic_num] = [i]
                else:
                    always_active.append(i)
        self.always_active = always_active
        self.atom_vocab_dict = atom_vocab_dict
        vocab_atom_dict = {}
        for k, v in atom_vocab_dict.items():
            for token in v:
                vocab_atom_dict[token] = k
        self.vocab_atom_dict = vocab_atom_dict

        vocab_vocab_dict = {}
        for k, v in atom_vocab_dict.items():
            for i in v:
                vocab_vocab_dict[i] = v
        for i in always_active:
            vocab_vocab_dict[i] = always_active
        self.vocab_vocab_dict = vocab_vocab_dict

    def get_valid_tokens_per_src_seq_in_batch(self, src):
        valid_tokens_per_seq = [
            np.unique([vocab_list for voc in np.unique(s.cpu().numpy()) for vocab_list in self.vocab_vocab_dict[voc]]) for s
            in src.t()]
        return valid_tokens_per_seq

    def get_log_probs_masking_matrix(self, src, beam_size):
        """
        Will make a matrix same beam * batch_size x vocab_size, where valid tokens are have entry 1 and other 1e-15.
        Therefore, if this multiplies the log_prob matrix, only valid tokens are predicted.
        """
        valid_tokens_per_seq = self.get_valid_tokens_per_src_seq_in_batch(src)
        mask = torch.stack([(torch.ones(len(self.vocab)).index_fill(0,
                            torch.tensor(valid_tokens), 0) * 1e15).index_fill(0, torch.tensor(valid_tokens), 1)
                            for valid_tokens in valid_tokens_per_seq for i in range(beam_size)])
        return mask

    def get_unique_vocab_counts_from_source(self, src):
        unique_counts_dicts = [dict(zip(*np.unique(s.cpu().numpy(), return_counts=True))) for s in src.t()]
        return unique_counts_dicts

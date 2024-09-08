import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNN(nn.Module):
    def __init__(self, config, name, input_shape, num_units, feature_num):
        super(FeatureNN, self).__init__()
        self.config = config
        self.name = name
        self.input_shape = input_shape
        self.num_units = num_units
        self.feature_num = feature_num
        self.fc = nn.Linear(input_shape, num_units)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return x

class NAM(nn.Module):
    def __init__(self, config, name, num_inputs: int, num_units: int) -> None:
        super(NAM, self).__init__()
        self._num_inputs = num_inputs
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.feature_nns = nn.ModuleList([
            FeatureNN(config=config, name=f'FeatureNN_{i}', input_shape=1, num_units=num_units, feature_num=i)
            for i in range(num_inputs)
        ])
        self.output_layer = nn.Linear(sum([num_units for _ in range(num_inputs)]), 3)
        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor):
        return [self.feature_nns[i](inputs[:, i:i+1]) for i in range(self._num_inputs)]

    def forward(self, inputs: torch.Tensor):
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout(conc_out)
        out = self.output_layer(dropout_out)
        return out, dropout_out

    def print_model_equation(self, feature_names):
        equation_terms = []
        feature_contributions = {}
        for i, fnn in enumerate(self.feature_nns):
            coefficients = fnn.fc.weight.data.flatten().tolist()
            intercepts = fnn.fc.bias.data.tolist()
            term = " + ".join([f"({coeff:.3f} * x_{feature_names[i]} + {intercept:.3f})" for coeff, intercept in zip(coefficients, intercepts)])
            equation_terms.append(term)
            feature_contributions[feature_names[i]] = sum(abs(c) for c in coefficients)
        equation = " + ".join(equation_terms) + f" + bias ({self._bias.item():.3f})"
        interpretability = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
        return interpretability[0][0], interpretability[-1][0]  # Return most and least contributing features

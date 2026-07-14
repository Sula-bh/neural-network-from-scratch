import numpy as np


def gradient_check(model, loss, inputs, targets, epsilon=1e-7, tolerance=1e-6):
    predictions = model.forward(inputs)
    loss.calculate(predictions, targets)
    model.backward(loss.backward())

    max_error = 0.0
    error_cases = []

    for layer in model.layers:
        if not layer.parameters:
            continue

        for param_idx in range(len(layer.parameters)):
            param = layer.parameters[param_idx]
            grad_backprop = layer.gradients[param_idx]

            it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

            while not it.finished:
                idx = it.multi_index
                original_value = param[idx]

                param[idx] = original_value + epsilon
                pred_plus = model.forward(inputs)
                loss_plus = loss.calculate(pred_plus, targets)

                param[idx] = original_value - epsilon
                pred_minus = model.forward(inputs)
                loss_minus = loss.calculate(pred_minus, targets)

                grad_approx = (loss_plus - loss_minus) / (2.0 * epsilon)

                param[idx] = original_value

                grad_true = grad_backprop[idx]

                numerator = np.abs(grad_true - grad_approx)
                denominator = np.maximum(np.abs(grad_true) + np.abs(grad_approx), 1e-15)
                relative_error = numerator / denominator

                if relative_error > tolerance:
                    error_cases.append(
                        {
                            "relative_error": float(relative_error),
                            "layer": layer.__class__.__name__,
                            "parameter_index": param_idx,
                            "index": idx,
                            "grad_true": float(grad_true),
                            "grad_approx": float(grad_approx),
                        }
                    )
                max_error = max(max_error, relative_error)
                it.iternext()

    return {
        "passed": not error_cases,
        "max_error": float(max_error),
        "error_cases": error_cases,
    }

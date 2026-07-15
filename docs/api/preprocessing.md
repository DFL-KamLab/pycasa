# API: Preprocessing

Preprocessing prepares loaded frames for downstream detection and tracking by generating stable visual representations: grayscale conversions, normalized intensity variants, and binary masks.

All preprocessing methods:
- Write derived arrays to `casa["video"]`.
- Update `casa["meta"]["last_preprocessing"]`.
- Return the same `Casa` instance for fluent chaining.

Methods are grouped into three namespaces: `self.preprocessing.grayscale(...)`, `self.preprocessing.normalization.*`, and `self.preprocessing.binarization.*`.

!!! note "This page is generated from the code"
    Signatures, parameters, and descriptions below are rendered directly from the
    `pycasa` source docstrings, so they always match the installed version.

---

## Grayscale

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionPreprocessingNamespace.grayscale

---

## Normalization

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionNormalizationNamespace.min_max

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionNormalizationNamespace.z_score

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionNormalizationNamespace.hist_equal

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionNormalizationNamespace.clahe

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionNormalizationNamespace.log

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionNormalizationNamespace.median

---

## Binarization

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionBinarizationNamespace.otsu

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionBinarizationNamespace.adaptive_mean

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionBinarizationNamespace.adaptive_gaussian

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionBinarizationNamespace.sauvola

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionBinarizationNamespace.niblack

::: pycasa.casa.preprocessing.preprocessing_wrapper._SessionBinarizationNamespace.urbano

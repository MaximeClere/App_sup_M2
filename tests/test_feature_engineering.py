"""Vérifications des transformations et imputations propres à chaque pli."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.preprocessing import FeatureEngineeringTransformer


class FeatureEngineeringTests(unittest.TestCase):
    """Vérifie les regroupements, imputations et schémas produits."""

    def _features(self) -> pd.DataFrame:
        """Crée des variables synthétiques avec des valeurs manquantes."""
        values = list(range(100))
        return pd.DataFrame(
            {
                "loan_to_value_ratio": [
                    np.nan if index == 0 else 60 + index * 0.5
                    for index in values
                ]
                + [600_000_000.0],
                "property_value": [
                    np.nan if index == 0 else 100_000 + index * 1_000
                    for index in values
                ]
                + [300_000],
                "income": [
                    np.nan if index == 0 else (-100 if index == 1 else 50 + index)
                    for index in values
                ]
                + [150],
                "loan_term": [np.nan if index == 0 else 360 for index in values]
                + [360],
                "total_units": [np.nan if index == 0 else 1 for index in values]
                + [2],
                "debt_to_income_ratio": (
                    ["20%-<30%"] * 40
                    + ["30%-<36%"] * 20
                    + ["49"] * 10
                    + ["<20%"] * 10
                    + ["50%-60%"] * 10
                    + [">60%"] * 10
                    + [pd.NA]
                ),
                "applicant_age": ["25-34"] * 80 + ["35-44"] * 20 + [pd.NA],
                "county_code": ["01001"] * 80 + ["01003"] * 20 + [""],
                "co-applicant_age": ["35-44"] * 101,
            }
        )

    def test_imputations_are_fit_on_training_rows_and_keep_schema(self) -> None:
        """Vérifie les statistiques pli-locales et le schéma commun des sorties."""
        training = self._features()
        training = pd.concat(
            [training, training.iloc[[1, 2]]], ignore_index=True
        )
        transformer = FeatureEngineeringTransformer().fit(training)
        transformed_training = transformer.transform(training)

        observed_ltv = training["loan_to_value_ratio"].dropna()
        expected_cap = observed_ltv.quantile(0.99)
        expected_mean = observed_ltv.clip(upper=expected_cap).mean()
        self.assertAlmostEqual(transformer.ltv_cap_, expected_cap)
        self.assertAlmostEqual(transformer.ltv_mean_, expected_mean)
        self.assertLess(transformer.ltv_cap_, observed_ltv.max())
        self.assertEqual(transformed_training.loc[0, "loan_to_value_ratio"], ">80–100")
        self.assertEqual(transformed_training.loc[40, "loan_to_value_ratio"], "≤80")
        self.assertEqual(transformed_training.loc[80, "loan_to_value_ratio"], ">80–100")
        self.assertEqual(transformed_training.loc[100, "loan_to_value_ratio"], ">100")
        self.assertEqual(transformed_training.loc[0, "debt_to_income_ratio"], "20%-49%")
        self.assertEqual(transformed_training.loc[100, "debt_to_income_ratio"], "20%-49%")
        self.assertEqual(transformed_training.loc[100, "applicant_age"], "25-34")
        self.assertEqual(transformed_training.loc[0, "loan_term"], 330)
        self.assertEqual(transformed_training.loc[0, "total_units"], 1)
        self.assertEqual(transformed_training.loc[1, "income"], -100)
        self.assertNotIn("co-applicant_age", transformed_training.columns)
        self.assertFalse(transformed_training.isna().any().any())

        validation = training.iloc[[0, 100]].copy()
        validation["loan_to_value_ratio"] = [np.nan, 600_000_000_000.0]
        transformed_validation = transformer.transform(validation)
        self.assertEqual(list(transformed_validation.columns), list(transformed_training.columns))
        self.assertAlmostEqual(transformer.ltv_mean_, expected_mean)


if __name__ == "__main__":
    unittest.main()

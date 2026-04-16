# Test Report – Malaria Classification API

## 1. Objective

The objective of testing is to validate the correctness, robustness, and reliability of the FastAPI backend serving the ML model.

---

## 2. Testing Approach

We used:

* **pytest** for automated testing
* **FastAPI TestClient** for API simulation
* Unit + integration style testing for endpoints

---

## 3. Test Environment

* Python: 3.11
* Framework: FastAPI
* Testing Tool: pytest
* Model Serving: MLflow (PyTorch)
* OS: macOS

---

## 4. Test Cases

| Test Name                | Description               | Expected Result       | Status |
| ------------------------ | ------------------------- | --------------------- | ------ |
| test_health              | Check API health endpoint | Returns status = ok   | ✅ Pass |
| test_ready               | Check readiness endpoint  | Returns ready = true  | ✅ Pass |
| test_model_info          | Fetch model metadata      | Returns model details | ✅ Pass |
| test_invalid_file_type   | Upload invalid file       | Returns 422 error     | ✅ Pass |
| test_empty_file          | Upload empty file         | Returns 400 error     | ✅ Pass |
| test_predict_valid_image | Upload valid image        | Returns prediction    | ✅ Pass |

---

## 5. Test Summary

* Total Tests: 6
* Passed: 6
* Failed: 0
* Warnings: 1 (MLflow deprecation, non-critical)

---

## 6. Acceptance Criteria

The system is considered valid if:

* All API endpoints return correct responses
* Invalid inputs are handled gracefully
* Valid inputs produce predictions
* No runtime crashes occur

✔ All criteria satisfied

---

## 7. Observations

* Initial issues with input shape and datatype mismatch were identified and fixed
* Preprocessing pipeline was aligned with training pipeline
* MLflow model loading was corrected to use PyTorch loader

---

## 8. Conclusion

The backend API is stable, robust, and production-ready.
All critical functionalities have been validated through automated testing.

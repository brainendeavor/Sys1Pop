use worker::*;

pub fn is_truthy(val: &str) -> bool {
    let lower = val.trim().to_lowercase();
    matches!(lower.as_str(), "true" | "1" | "yes" | "on")
}

pub fn check_token_match(expected: &str, auth_header: Option<&str>, x_token: Option<&str>) -> bool {
    let expected = expected.trim();
    if expected.is_empty() {
        return true;
    }

    let token = auth_header
        .and_then(|h| h.strip_prefix("Bearer ").map(|s| s.trim()))
        .or_else(|| x_token.map(|s| s.trim()));

    token == Some(expected)
}

pub fn is_admin_disabled(val: Option<&str>) -> bool {
    match val {
        Some(v) => {
            let lower = v.trim().to_lowercase();
            matches!(lower.as_str(), "false" | "0" | "no" | "off")
        }
        None => false,
    }
}

pub fn get_env_var_trimmed(env: &Env, name: &str) -> Option<String> {
    env.var(name)
        .or_else(|_| env.secret(name))
        .map(|v| v.to_string().trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
}

pub fn is_secure_decide_api_enabled(env: &Env) -> bool {
    for key in ["SECURE_DECIDE_API", "SECURE_ALL_APIS", "REQUIRE_AUTH"] {
        if let Some(val) = get_env_var_trimmed(env, key) {
            if is_truthy(&val) {
                return true;
            }
        }
    }
    false
}

pub fn verify_api_token(req: &Request, env: &Env) -> Result<Option<Response>> {
    let expected = get_env_var_trimmed(env, "API_TOKEN");

    match expected {
        Some(expected_token) => {
            let headers = req.headers();
            let auth_header = headers.get("Authorization").ok().flatten();
            let x_token = headers.get("X-API-Token").ok().flatten();

            if check_token_match(&expected_token, auth_header.as_deref(), x_token.as_deref()) {
                Ok(None)
            } else {
                Ok(Some(Response::error(
                    "Unauthorized: Invalid or missing API token",
                    401,
                )?))
            }
        }
        None => {
            if is_secure_decide_api_enabled(env) {
                Ok(Some(Response::error(
                    "Unauthorized: SECURE_DECIDE_API is enabled but API_TOKEN is not configured on worker",
                    401,
                )?))
            } else {
                Ok(None)
            }
        }
    }
}

pub fn verify_admin_auth(req: &Request, env: &Env) -> Result<Option<Response>> {
    let admin_enabled_var = get_env_var_trimmed(env, "ENABLE_ADMIN_API");
    if is_admin_disabled(admin_enabled_var.as_deref()) {
        return Ok(Some(Response::error("Admin lifecycle API is disabled", 403)?));
    }

    verify_api_token(req, env)
}

pub fn verify_decide_auth(req: &Request, env: &Env) -> Result<Option<Response>> {
    if is_secure_decide_api_enabled(env) {
        verify_api_token(req, env)
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_truthy() {
        assert!(is_truthy("true"));
        assert!(is_truthy("TRUE"));
        assert!(is_truthy("1"));
        assert!(is_truthy("yes"));
        assert!(is_truthy("on"));
        assert!(!is_truthy("false"));
        assert!(!is_truthy("0"));
        assert!(!is_truthy("no"));
        assert!(!is_truthy("off"));
        assert!(!is_truthy("random"));
    }

    #[test]
    fn test_is_admin_disabled() {
        assert!(is_admin_disabled(Some("false")));
        assert!(is_admin_disabled(Some("0")));
        assert!(is_admin_disabled(Some("no")));
        assert!(is_admin_disabled(Some("off")));
        assert!(!is_admin_disabled(Some("true")));
        assert!(!is_admin_disabled(Some("1")));
        assert!(!is_admin_disabled(None));
    }

    #[test]
    fn test_check_token_match_bearer() {
        let expected = "my-secret-token";
        assert!(check_token_match(expected, Some("Bearer my-secret-token"), None));
        assert!(check_token_match(expected, Some("Bearer  my-secret-token "), None));
        assert!(!check_token_match(expected, Some("Bearer wrong-token"), None));
        assert!(!check_token_match(expected, Some("Basic dXNlcjpwYXNz"), None));
        assert!(!check_token_match(expected, None, None));
    }

    #[test]
    fn test_check_token_match_x_api_token() {
        let expected = "my-secret-token";
        assert!(check_token_match(expected, None, Some("my-secret-token")));
        assert!(check_token_match(expected, None, Some(" my-secret-token ")));
        assert!(!check_token_match(expected, None, Some("wrong-token")));
    }

    #[test]
    fn test_check_token_match_empty_expected() {
        assert!(check_token_match("", None, None));
        assert!(check_token_match("   ", None, None));
    }
}

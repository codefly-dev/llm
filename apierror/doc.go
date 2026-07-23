// Package apierror normalizes transport-facing API errors.
//
// It provides small helpers for mapping internal failures to stable external
// error shapes. Domain packages should return contextual errors; service
// handlers use this package when crossing the API boundary.
package apierror

package contract

import "testing"

func TestParseModelFamilyIsNormalizedAndFailClosed(t *testing.T) {
	family, err := ParseModelFamily(" OpenAI ")
	if err != nil || family != ModelFamilyOpenAI {
		t.Fatalf("family = %q, err = %v", family, err)
	}
	if _, err := ParseModelFamily("opneai"); err == nil {
		t.Fatal("model-family typo must not fall through to a default")
	}
}

package dev.yourorg.tools.japicmp_analyzer;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import japicmp.model.JApiCompatibilityChange;

import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Objects;

public class ApiChange {
    private int modifier;
    private String returnType;
    private String element;
    private ApiChangeType action;
    private String longName;
    private List<JApiCompatibilityChange> category;
    private String description;
    private String name;
    private ApiMetadata newVersion;
    private ApiMetadata oldVersion;

    public ApiChange() {
    }

    public int getModifier() {
        return modifier;
    }

    public ApiChange setModifier(int modifier) {
        this.modifier = modifier;
        return this;
    }

    public String getReturnType() {
        return returnType;
    }

    public ApiChange setReturnType(String returnType) {
        this.returnType = returnType;
        return this;
    }

    public String getElement() {
        return element;
    }

    public ApiChange setElement(String element) {
        this.element = element;
        return this;
    }

    public ApiChangeType getAction() {
        return action;
    }

    public ApiChange setAction(ApiChangeType action) {
        this.action = action;
        return this;
    }

    public String getLongName() {
        return longName;
    }

    public ApiChange setLongName(String longName) {
        this.longName = longName;
        return this;
    }

    public List<JApiCompatibilityChange> getCategory() {
        return category;
    }

    public ApiChange setCategory(List<JApiCompatibilityChange> category) {
        this.category = category;
        return this;
    }

    public String getDescription() {
        return description;
    }

    public ApiChange setDescription(String description) {
        this.description = description;
        return this;
    }

    public String getName() {
        return name;
    }

    public ApiChange setName(String name) {
        this.name = name;
        return this;
    }

    public ApiMetadata getNewVersion() {
        return newVersion;
    }

    public ApiChange setNewVersion(ApiMetadata newVersion) {
        this.newVersion = newVersion;
        return this;
    }

    public ApiMetadata getOldVersion() {
        return oldVersion;
    }

    public ApiChange setOldVersion(ApiMetadata oldVersion) {
        this.oldVersion = oldVersion;
        return this;
    }

    @Override
    public String toString() {
        ObjectMapper mapper = new ObjectMapper();
        try {
            return mapper.writeValueAsString(this);
        } catch (JsonProcessingException e) {
            return "";
        }
    }

    public String toDiffString() {
        if (ApiChangeType.REMOVE.equals(this.action)) {
            return "-- " + this.getCompleteValue();
        } else {
            return "++ " + this.getCompleteValue();
        }
    }

    public String getValue() {
        return this.element;
    }

    public String getCompleteValue() {
        return Modifier.toString(this.modifier) + " " + this.returnType + " " + this.element;
    }

    public boolean isSame(ApiChange apiChange) {
        return this.getCompleteValue().equals(apiChange.getCompleteValue());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ApiChange that = (ApiChange) o;
        return modifier == that.modifier
                && Objects.equals(returnType, that.returnType)
                && Objects.equals(element, that.element)
                && Objects.equals(action, that.action)
                && Objects.equals(longName, that.longName)
                && Objects.equals(category, that.category)
                && Objects.equals(description, that.description)
                && Objects.equals(name, that.name)
                && Objects.equals(newVersion, that.newVersion)
                && Objects.equals(oldVersion, that.oldVersion);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modifier, returnType, element, action, longName, category, description, name, newVersion, oldVersion);
    }
}

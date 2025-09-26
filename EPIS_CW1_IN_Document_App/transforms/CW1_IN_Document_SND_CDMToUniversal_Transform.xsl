```xsl
<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xs="http://www.w3.org/2001/XMLSchema"
                xmlns:cdm="http://example.com/cdm"
                xmlns:ee="http://example.com/ee"
                exclude-result-prefixes="xs cdm ee">

    <!--
        Transformation Name: CDM_DocumentMessage_To_EE_UniversalEvent
        Purpose: Transforms CDM Document Message to Universal Event
        Type: XSL Transform
    -->

    <!-- Define the output method and encoding -->
    <xsl:output method="xml" encoding="UTF-8" indent="yes"/>

    <!-- Define the template to match the root element of the source document -->
    <xsl:template match="/cdm:DocumentMessage">
        <!-- Create the root element of the target document -->
        <ee:UniversalEvent>
            <!-- Apply templates to transform child elements -->
            <xsl:apply-templates select="*"/>
        </ee:UniversalEvent>
    </xsl:template>

    <!-- Define templates to transform child elements -->
    <!-- Add templates here to implement field mapping requirements -->

    <!-- Template to handle missing or invalid data -->
    <xsl:template match="*[not(text())]">
        <!-- Log an error message for missing or invalid data -->
        <xsl:message>Missing or invalid data: <xsl:value-of select="name()"/></xsl:message>
        <!-- Create an empty element to represent the missing or invalid data -->
        <xsl:element name="{local-name()}"/>
    </xsl:template>

    <!-- Template to handle business rule validation and conditional logic -->
    <xsl:template match="*[text()]">
        <!-- Apply business rule validation and conditional logic here -->
        <!-- Use xsl:choose, xsl:when, and xsl:otherwise to implement conditional logic -->
        <xsl:choose>
            <!-- Add conditions here to implement business rules -->
            <xsl:when test="condition1">
                <!-- Transform the element based on condition1 -->
                <xsl:element name="{local-name()}">
                    <xsl:value-of select="text()"/>
                </xsl:element>
            </xsl:when>
            <xsl:otherwise>
                <!-- Transform the element based on default condition -->
                <xsl:element name="{local-name()}">
                    <xsl:value-of select="text()"/>
                </xsl:element>
            </xsl:otherwise>
        </xsl:choose>
    </xsl:template>

    <!-- Define a template to handle complex data conversions -->
    <xsl:template match="*[contains(local-name(), 'Date')]">
        <!-- Use the xs:dateTime function to convert the date string to a dateTime object -->
        <xsl:element name="{local-name()}">
            <xsl:value-of select="xs:dateTime(text())"/>
        </xsl:element>
    </xsl:template>

    <!-- Define a template to handle simple field mappings -->
    <xsl:template match="*[not(contains(local-name(), 'Date'))]">
        <!-- Use the xsl:value-of element to copy the text content of the element -->
        <xsl:element name="{local-name()}">
            <xsl:value-of select="text()"/>
        </xsl:element>
    </xsl:template>

</xsl:stylesheet>
```